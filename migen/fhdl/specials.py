from operator import itemgetter

from migen.fhdl.structure import *
from migen.fhdl.structure import _Value
from migen.fhdl.bitcontainer import bits_for, value_bits_sign
from migen.fhdl.tools import *
from migen.fhdl.tracer import get_obj_var_name
from migen.fhdl.verilog import _printexpr as verilog_printexpr
from migen.fhdl import vhdl

__all__ = ["TSTriple", "Instance", "Memory",
    "READ_FIRST", "WRITE_FIRST", "NO_CHANGE"]


class Special(DUID):
    def iter_expressions(self):
        for x in []:
            yield x

    def rename_clock_domain(self, old, new):
        for obj, attr, direction in self.iter_expressions():
            rename_clock_domain_expr(getattr(obj, attr), old, new)

    def list_clock_domains(self):
        r = set()
        for obj, attr, direction in self.iter_expressions():
            r |= list_clock_domains_expr(getattr(obj, attr))
        return r

    def list_ios(self, ins, outs, inouts):
        r = set()
        for obj, attr, direction in self.iter_expressions():
            if (direction == SPECIAL_INPUT and ins) \
              or (direction == SPECIAL_OUTPUT and outs) \
              or (direction == SPECIAL_INOUT and inouts):
                signals = list_signals(getattr(obj, attr))
                r.update(signals)
        return r


class Tristate(Special):
    def __init__(self, target, o, oe, i=None):
        Special.__init__(self)
        self.target = wrap(target)
        self.o = wrap(o)
        self.oe = wrap(oe)
        self.i = wrap(i) if i is not None else None

    def iter_expressions(self):
        for attr, target_context in [
          ("target", SPECIAL_INOUT),
          ("o", SPECIAL_INPUT),
          ("oe", SPECIAL_INPUT),
          ("i", SPECIAL_OUTPUT)]:
            if getattr(self, attr) is not None:
                yield self, attr, target_context

    @staticmethod
    def emit_verilog(tristate, ns, add_data_file):
        def pe(e):
            return verilog_printexpr(ns, e)[0]
        w, s = value_bits_sign(tristate.target)
        r = "assign " + pe(tristate.target) + " = " \
            + pe(tristate.oe) + " ? " + pe(tristate.o) \
            + " : " + str(w) + "'bz;\n"
        if tristate.i is not None:
            r += "assign " + pe(tristate.i) + " = " + pe(tristate.target) + ";\n"
        r += "\n"
        return r


class TSTriple:
    def __init__(self, bits_sign=None, min=None, max=None, reset_o=0, reset_oe=0):
        self.o = Signal(bits_sign, min=min, max=max, reset=reset_o)
        self.oe = Signal(reset=reset_oe)
        self.i = Signal(bits_sign, min=min, max=max)

    def get_tristate(self, target):
        return Tristate(target, self.o, self.oe, self.i)


class Instance(Special):
    class _IO:
        def __init__(self, name, expr=None):
            self.name = name
            if expr is None:
                expr = Signal()
            self.expr = wrap(expr)
    class Input(_IO):
        pass
    class Output(_IO):
        pass
    class InOut(_IO):
        pass
    class Parameter:
        def __init__(self, name, value):
            self.name = name
            if isinstance(value, (int, bool)):
                value = Constant(value)
            self.value = value
    class PreformattedParam(str):
        pass

    def __init__(self, of, *items, name="", synthesis_directive=None, **kwargs):
        Special.__init__(self)
        self.of = of
        if name:
            self.name_override = name
        else:
            self.name_override = of
        self.items = list(items)
        self.synthesis_directive = synthesis_directive
        for k, v in sorted(kwargs.items(), key=itemgetter(0)):
            try:
                item_type, item_name = k.split("_", maxsplit=1)
            except ValueError:
                raise TypeError("Wrong format for value '" + str(k) +
                                "', format should be 'type_name'")

            item_class = {
                "i": Instance.Input,
                "o": Instance.Output,
                "io": Instance.InOut,
                "p": Instance.Parameter
            }[item_type]
            self.items.append(item_class(item_name, v))

    def get_io(self, name):
        for item in self.items:
            if isinstance(item, Instance._IO) and item.name == name:
                return item.expr

    def iter_expressions(self):
        for item in self.items:
            if isinstance(item, Instance.Input):
                yield item, "expr", SPECIAL_INPUT
            elif isinstance(item, Instance.Output):
                yield item, "expr", SPECIAL_OUTPUT
            elif isinstance(item, Instance.InOut):
                yield item, "expr", SPECIAL_INOUT

    @staticmethod
    def emit_verilog(instance, ns, add_data_file):
        r = instance.of + " "
        parameters = list(filter(lambda i: isinstance(i, Instance.Parameter), instance.items))
        if parameters:
            r += "#(\n"
            firstp = True
            for p in parameters:
                if not firstp:
                    r += ",\n"
                firstp = False
                r += "\t." + p.name + "("
                if isinstance(p.value, Constant):
                    r += verilog_printexpr(ns, p.value)[0]
                elif isinstance(p.value, float):
                    r += str(p.value)
                elif isinstance(p.value, Instance.PreformattedParam):
                    r += p.value
                elif isinstance(p.value, str):
                    r += "\"" + p.value + "\""
                else:
                    raise TypeError
                r += ")"
            r += "\n) "
        r += ns.get_name(instance)
        if parameters: r += " "
        r += "(\n"
        firstp = True
        for p in instance.items:
            if isinstance(p, Instance._IO):
                name_inst = p.name
                name_design = verilog_printexpr(ns, p.expr)[0]
                if not firstp:
                    r += ",\n"
                firstp = False
                r += "\t." + name_inst + "(" + name_design + ")"
        if not firstp:
            r += "\n"
        if instance.synthesis_directive is not None:
            synthesis_directive = "/* synthesis {} */".format(instance.synthesis_directive)
            r += ")" + synthesis_directive + ";\n\n"
        else:
            r += ");\n\n"
        return r


(READ_FIRST, WRITE_FIRST, NO_CHANGE) = range(3)


class _MemoryPort(Special):
    def __init__(self, adr, dat_r, we=None, dat_w=None,
      async_read=False, re=None, we_granularity=0, mode=WRITE_FIRST,
      clock_domain="sys"):
        Special.__init__(self)
        self.adr = adr
        self.dat_r = dat_r
        self.we = we
        self.dat_w = dat_w
        self.async_read = async_read
        self.re = re
        self.we_granularity = we_granularity
        self.mode = mode
        self.clock = ClockSignal(clock_domain)

    def iter_expressions(self):
        for attr, target_context in [
          ("adr", SPECIAL_INPUT),
          ("we", SPECIAL_INPUT),
          ("dat_w", SPECIAL_INPUT),
          ("re", SPECIAL_INPUT),
          ("dat_r", SPECIAL_OUTPUT),
          ("clock", SPECIAL_INPUT)]:
            yield self, attr, target_context

    @staticmethod
    def emit_vhdl(port, converter, ns, add_data_file):
        return {}  # done by parent Memory object

    @staticmethod
    def emit_verilog(port, ns, add_data_file):
        return ""  # done by parent Memory object


class _MemoryLocation(_Value):
    def __init__(self, memory, index):
        _Value.__init__(self)
        self.memory = memory
        self.index = wrap(index)


class Memory(Special):
    def __init__(self, width, depth, init=None, name=None):
        Special.__init__(self)
        self.width = width
        self.depth = depth
        self.ports = []
        self.init = init
        self.name_override = get_obj_var_name(name, "mem")

    def __getitem__(self, index):
        # simulation only
        return _MemoryLocation(self, index)

    def get_port(self, write_capable=False, async_read=False,
      has_re=False, we_granularity=0, mode=WRITE_FIRST,
      clock_domain="sys"):
        if we_granularity >= self.width:
            we_granularity = 0
        adr = Signal(max=self.depth)
        dat_r = Signal(self.width)
        if write_capable:
            if we_granularity:
                we = Signal(self.width//we_granularity)
            else:
                we = Signal()
            dat_w = Signal(self.width)
        else:
            we = None
            dat_w = None
        if has_re:
            re = Signal()
        else:
            re = None
        mp = _MemoryPort(adr, dat_r, we, dat_w,
          async_read, re, we_granularity, mode,
          clock_domain)
        self.ports.append(mp)
        return mp

    @staticmethod
    def emit_vhdl(memory, converter, ns, add_data_file):
        def pex(e, type=None):
            return converter._printexpr(ns, e, type)
        use = []

        decl = ""
        decl += "signal "+ns.get_name(memory)+": array(0 to "+str(memory.depth-1)+') of std_logic_vector;\n'

        body = ""
        name = ns.get_name(memory)
        adrbits = bits_for(memory.depth-1)

        adr_regs = {}
        data_regs = {}
        for port in memory.ports:
            if not port.async_read:
                if port.mode == WRITE_FIRST and port.we is not None:
                    adr_reg = Signal(adrbits,name_override="memadr")
                    decl += "signal "+ns.get_name(adr_reg)+ ": unsigned(" + str(adrbits-1) + " downto 0);\n"
                    adr_regs[id(port)] = adr_reg
                else:
                    data_reg = Signal(memory.width,name_override="memdat")
                    decl += "signal "+ns.get_name(data_reg)+ ": unsigned(" + str(memory.width-1) + " downto 0);\n"
                    data_regs[id(port)] = data_reg

        for k,port in enumerate(memory.ports):
#            adr = pex(port.adr,vhdl.integer.constrain(0,memory.depth-1))
            adr = pex(port.adr,vhdl.unsigned[adrbits-1:0])
            if port.we is not None or not port.async_read:
                clk = pex(port.clock)
                body += "ram_{name}_port_{pidx}: process({clk}) is\nbegin\nif rising_edge({clk}) then\n".format(name=name,pidx=k,clk=clk)
                if port.we is not None:
                    dat_w = pex(port.dat_w,vhdl.unsigned[memory.width-1:0])
                    if port.we_granularity:
                        n = memory.width//port.we_granularity
                        for i in range(n):
                            m = i*port.we_granularity
                            M = (i+1)*port.we_granularity-1
                            sl = "(" + str(M) + " downto " + str(m) + ")"
                            body += "\tif " + pex(port.we,vhdl.std_logic_vector[n-1:0]) + "(" + str(i) + ") = '1' then \n"
                            body += "\t\t" + name + "(" + adr + ")" + sl + " <= " + dat_w + sl + ";\nend if;\n"
                    else:
                        body += "\tif " + pex(port.we,vhdl.boolean) + " then\n"
                        body += "\t\t" + name + "(" + adr + ") <= " + dat_w + ";\nend if\n"
                if not port.async_read:
                    if port.mode == WRITE_FIRST and port.we is not None:
                        rd = "\t" + pex(adr_regs[id(port)]) + " <= " + adr + ";\n"
                    else:
                        bassign = pex(data_regs[id(port)]) + " <= " + name + "(" + adr + ");\n"   # TODO: ensure types match
                        if port.mode == READ_FIRST or port.we is None:
                            rd = "\t" + bassign
                        elif port.mode == NO_CHANGE:
                            rd = "\tif not " + pex(port.we,vhdl.boolean) + " then\n" \
                              + "\t\t" + bassign + 'end if;\n'
                    if port.re is None:
                        body += rd
                    else:
                        body += "\tif " + pex(port.re, vhdl.boolean) + " then \n"
                        body += "\t" + rd.replace("\n\t", "\n\t\t") + "\tend if;\n"
                body += "end if;\nend process;\n"
            else:
                body += "-- ram {name} port {pidx}\n".format(name=name,pidx=k)

            dat_r = pex(port.dat_r)
            if port.async_read:
                body += dat_r + " <= " + name + "(" + adr + ");\n"
            else:
                if port.mode == WRITE_FIRST and port.we is not None:
                    body += dat_r + " <= " + name + "(" + pex(adr_regs[id(port)],vhdl.integer) + ");\n"
                else:
                    body += dat_r + " <= " + pex(data_regs[id(port)],converter.typeof(port.dat_r)) + ";\n"
            body += "\n"

        if memory.init is not None:
            raise NotImplementedError('initialising memory')

        return dict(use=use,decl=decl,body=body)

    @staticmethod
    def emit_verilog(memory, ns, add_data_file):
        r = ""
        def gn(e):
            if isinstance(e, Memory):
                return ns.get_name(e)
            else:
                return verilog_printexpr(ns, e)[0]
        adrbits = bits_for(memory.depth-1)

        r += "reg [" + str(memory.width-1) + ":0] " \
            + gn(memory) \
            + "[0:" + str(memory.depth-1) + "];\n"

        adr_regs = {}
        data_regs = {}
        for port in memory.ports:
            if not port.async_read:
                if port.mode == WRITE_FIRST and port.we is not None:
                    adr_reg = Signal(name_override="memadr")
                    r += "reg [" + str(adrbits-1) + ":0] " \
                        + gn(adr_reg) + ";\n"
                    adr_regs[id(port)] = adr_reg
                else:
                    data_reg = Signal(name_override="memdat")
                    r += "reg [" + str(memory.width-1) + ":0] " \
                        + gn(data_reg) + ";\n"
                    data_regs[id(port)] = data_reg

        for port in memory.ports:
            r += "always @(posedge " + gn(port.clock) + ") begin\n"
            if port.we is not None:
                if port.we_granularity:
                    n = memory.width//port.we_granularity
                    for i in range(n):
                        m = i*port.we_granularity
                        M = (i+1)*port.we_granularity-1
                        sl = "[" + str(M) + ":" + str(m) + "]"
                        r += "\tif (" + gn(port.we) + "[" + str(i) + "])\n"
                        r += "\t\t" + gn(memory) + "[" + gn(port.adr) + "]" + sl + " <= " + gn(port.dat_w) + sl + ";\n"
                else:
                    r += "\tif (" + gn(port.we) + ")\n"
                    r += "\t\t" + gn(memory) + "[" + gn(port.adr) + "] <= " + gn(port.dat_w) + ";\n"
            if not port.async_read:
                if port.mode == WRITE_FIRST and port.we is not None:
                    rd = "\t" + gn(adr_regs[id(port)]) + " <= " + gn(port.adr) + ";\n"
                else:
                    bassign = gn(data_regs[id(port)]) + " <= " + gn(memory) + "[" + gn(port.adr) + "];\n"
                    if port.mode == READ_FIRST or port.we is None:
                        rd = "\t" + bassign
                    elif port.mode == NO_CHANGE:
                        rd = "\tif (!" + gn(port.we) + ")\n" \
                          + "\t\t" + bassign
                if port.re is None:
                    r += rd
                else:
                    r += "\tif (" + gn(port.re) + ")\n"
                    r += "\t" + rd.replace("\n\t", "\n\t\t")
            r += "end\n\n"

        for port in memory.ports:
            if port.async_read:
                r += "assign " + gn(port.dat_r) + " = " + gn(memory) + "[" + gn(port.adr) + "];\n"
            else:
                if port.mode == WRITE_FIRST and port.we is not None:
                    r += "assign " + gn(port.dat_r) + " = " + gn(memory) + "[" + gn(adr_regs[id(port)]) + "];\n"
                else:
                    r += "assign " + gn(port.dat_r) + " = " + gn(data_regs[id(port)]) + ";\n"
        r += "\n"

        if memory.init is not None:
            content = ""
            for d in memory.init:
                content += "{:x}\n".format(d)
            memory_filename = add_data_file(gn(memory) + ".init", content)

            r += "initial begin\n"
            r += "\t$readmemh(\"" + memory_filename + "\", " + gn(memory) + ");\n"
            r += "end\n\n"

        return r


class SynthesisDirective(Special):
    def __init__(self, template, **signals):
        Special.__init__(self)
        self.template = template
        self.signals = signals

    @staticmethod
    def emit_verilog(directive, ns, add_data_file):
        name_dict = dict((k, ns.get_name(sig)) for k, sig in directive.signals.items())
        formatted = directive.template.format(**name_dict)
        return "// synthesis " + formatted + "\n"


class Keep(SynthesisDirective):
    def __init__(self, signal):
        SynthesisDirective.__init__(self, "attribute keep of {s} is true", s=signal)
