from itertools import count
import tempfile
import os
from collections import OrderedDict
import shutil
import datetime

from migen.fhdl.namer import build_namespace


def vcd_codes():
    codechars = [chr(i) for i in range(33, 127)]
    for n in count():
        q, r = divmod(n, len(codechars))
        code = codechars[r]
        while q > 0:
            q, r = divmod(q, len(codechars))
            code = codechars[r] + code
        yield code


class VCDWriter:
    def __init__(self, filename):
        self.filename = filename
        self.buffer_file = tempfile.TemporaryFile(
            dir=os.path.dirname(filename), mode="w+")
        self.codegen = vcd_codes()
        self.codes = OrderedDict()
        self.signal_values = dict()
        self.t = 0

    def _write_value(self, f, signal, value):
        l = len(signal)
        if value < 0:
            value += 2**l
        if l > 1:
            fmtstr = "b{:0" + str(l) + "b} {}\n"
        else:
            fmtstr = "{}{}\n"
        try:
            code = self.codes[signal]
        except KeyError:
            code = next(self.codegen)
            self.codes[signal] = code
        f.write(fmtstr.format(value, code))

    def set(self, signal, value):
        if (signal not in self.signal_values
                or self.signal_values[signal] != value):
            self._write_value(self.buffer_file, signal, value)
            self.signal_values[signal] = value

    def delay(self, delay):
        self.t += delay
        self.buffer_file.write("#{}\n".format(self.t))

    def close(self):
        from migen import __version__
        with open(self.filename, "w") as out:
            out.write('$date %s $end\n'%str(datetime.datetime.now()))
            out.write('$version Generated by Migen %s $end\n'%__version__)
            ns = build_namespace(self.codes.keys())
            for signal, code in self.codes.items():
                name = ns.get_name(signal)
                out.write("$var wire {len} {code} {name} $end\n"
                          .format(name=name, code=code, len=len(signal)))
            out.write("$enddefinitions $end\n$dumpvars\n")
            for signal in self.codes.keys():
                self._write_value(out, signal, signal.reset.value)
            out.write("$end\n")
            out.write("#0\n")

            self.buffer_file.seek(0)
            shutil.copyfileobj(self.buffer_file, out)
            self.buffer_file.close()

class DummyVCDWriter:
    def set(self, signal, value):
        pass

    def delay(self, delay):
        pass

    def close(self):
        pass

def VCD_events(fh):
    """ iterates change events in a VCD file

    """
    signals = {}
    level = []
    enddefinitions = False

    def tokenize(src):
        for line in src:
            yield from line.split()
    stream = tokenize(fh)

    for token in stream:
        if token.startswith('$'):
            # a section
            token = token[1:]
            content = []
            for t in stream:
                if t == '$end':
                    break
                content.append(t)
            else:
                raise ValueError('Missing $end')

            if token == "enddefinitions":
                enddefinitions = True
                yield signals

            elif token == "timescale":
                timescale = ''.join(content)

            elif token == "scope":
                type, name = content
                level.append(name)

            elif token == "upscope":
                assert not content
                level.pop()

            elif token == "var":
                assert not enddefinitions, "Unexpected variable definition after $enddefinitions"
                type,width,code = content[:3]
                name = ''.join(content[3:])
                path = '.'.join(level)
                full_name = path + ('.' if path else '') + name
                signals.setdefault(code,[]).append({
                    'type': type,
                    'name': name,
                    'width': int(width),
                    'hier': path,
                    'full_name': full_name,
                })

            elif token in {'version','date','comment'}:
                # ignore them
                pass

            elif token in {'dumpall','dumpon','dumpoff','dumpvars'}:
                # ignore them as well
                pass

            else:
                # print('Ignoring unknown section ',token,content)
                pass
            continue

        assert enddefinitions, "Expected $enddefinitions before any value change"

        if token.startswith('#'):
            time = int(token[1:])
            yield time

        elif token[0] in '01xzXZ':
            val = token[0] == '1'
            code = token[1:]
            yield code,val

        elif token[0] in 'b':
            val = int(token[1:],base=2)
            code = next(stream)
            yield code, val

        elif token[0] in 'r':
            val = float(token[1:])
            code = next(stream)
            yield code, val

        else:
            raise ValueError('Unknown token "%s"'%token)

class NumpySignalTrace:
    def __init__(self,type,width):
        import numpy as np
        dtype = bool if width == 1 else float if width=='real' else int
        self.size = 0
        self.type = type
        self.width = width
        self._time = np.empty(0,'i8')
        self._value = np.empty(0,dtype)

    def append(self,time,value):
        k = self.size
        self._time = self._insert(self._time,k,time)
        self._value = self._insert(self._value,k,value)
        self.size += 1

    def _insert(self,array,idx,val):
        import numpy as np
        if idx>=array.size:
            array = np.resize(array,2*max(idx,16))
        array[idx] = val
        return array

    @property
    def time(self):
        return self._time[:self.size]

    @property
    def value(self):
        return self._value[:self.size]

def dump_VCD_events(src):
    descr = next(src)
    signals = {}
    traces = {}
    time = None
    for k,v in descr.items():
        v0 = v[0]
        type = v0['type']
        width = v0['width']
        trace = NumpySignalTrace(type,width)
        traces[k] = trace
        for vn in v:
            assert vn['type'] == type
            assert vn['width'] == width
            n = vn['full_name']
            assert not n in signals
            signals[n] = trace
    for e in src:
        if isinstance(e,int):
            time = e
            continue
        if isinstance(e,tuple):
            code, val = e
            traces[code].append(time, val)
            continue
    return signals
