from functools import partial
from operator import itemgetter
import abc, itertools
import collections
import abc

from migen.fhdl.structure import *
from migen.fhdl.structure import _Operator, _Slice, _Assign, _Fragment, _Value, _Statement
from migen.fhdl.tools import *
from migen.fhdl.visit import NodeTransformer, ContextualTransformer, visitor_for
from migen.fhdl.namer import build_namespace
from migen.fhdl.conv_output import ConvOutput
from migen.fhdl.bitcontainer import value_bits_sign

_reserved_keywords = {
    "abs", "access", "after", "alias", "all", "and", "architecture", "array", "assert",
    "attribute", "begin", "block", "body", "buffer", "bus", "case", "component",
    "configuration", "constant", "disconnect", "downto", "else", "elsif", "end",
    "entity", "exit", "file", "for", "function", "generate", "generic", "group",
    "guarded", "if", "impure", "in", "inertial", "inout", "is", "label", "library",
    "linkage", "literal", "loop", "map", "mod", "nand", "new", "next", "nor", "not",
    "null", "of", "on", "open", "or", "others", "out", "package", "port", "postponed",
    "procedure", "process", "pure", "range", "record", "register", "reject", "return",
    "rol", "ror", "select", "severity", "shared", "signal", "sla", "sli", "sra", "srl",
    "subtype", "then", "to", "transport", "type", "unaffected", "units", "until"
}

# -------------
# Simple type system for Migen
# -------------

class MigenExpressionType(abc.ABC):
    """ Abstract base class for Migen's type hierarchy.

    It describes the semantics of the type:
    1. The set of allowed values (i.e. for integers it's range)
    2. The behaviour of operations performed on the type (Migen-level operator overloading)
    """
    @abc.abstractproperty
    def size(self):
        """ The number of bits required to completely represent the type.
        """

class MigenInteger(MigenExpressionType):
    def __init__(self,width):
        self._width = width

    @property
    def size(self):
        return self._width

    @abc.abstractproperty
    def min(self):
        """ Lowest possible value.

        The allowed range is [min .. max] with both ends inclusive.
        """

    @abc.abstractproperty
    def max(self):
        """ Highest possible value.

        The allowed range is [min .. max] with both ends inclusive.
        """

class Unsigned(MigenInteger):
    @property
    def min(self): return 0
    @property
    def max(self): return (1<<self._width) - 1

class Signed(MigenInteger):
    @property
    def min(self): return -1<<(self._width-1)
    @property
    def max(self): return (1<<(self._width-1)) - 1


# -------------
# VHDL types
# -------------

class VHDLType(abc.ABC):
    _identity = ('name',)

    def __eq__(self,other):
        if not isinstance(other,VHDLType):
            return NotImplemented
        return type(other) == type(self) and self._prehash == other._prehash

    @property
    def _prehash(self):
        return tuple(getattr(self,attr) for attr in type(self)._identity)

    def __hash__(self):
        return hash(self._prehash)

    @abc.abstractmethod
    def equivalent_to(self,other):
        pass

    @abc.abstractmethod
    def castable_to(self,other):
        pass

    def compatible_with(self,other):
        """ Can be assigned without a cast. """
        return self.ultimate_base == other.ultimate_base

    unconstrained = False

    @property
    def ultimate_base(self):
        return self

class VHDLSubtype(abc.ABC):
    """ Mixin for subtypes
    """
    def __init__(self,base,name=None,*a,**kw):
        self.base = base
        super(VHDLSubtype,self).__init__(name,*a,**kw)

    @property
    def ultimate_base(self):
        return self.base.ultimate_base

class VHDLAccess(VHDLType):
    pass
class VHDLFile(VHDLType):
    pass

class VHDLScalar(VHDLType):
    @abc.abstractproperty
    def length(self):
        pass

class VHDLInteger(VHDLScalar):
    _identity = ('name','left','right','ascending')
    def __init__(self,name,left=None,right=None,ascending=None):
        assert (left is None) == (right is None)
        if ascending is None:
            if left is not None:
                ascending = not left>right
        else:
            assert left is not None
            assert left == right or ascending == (left<right)
        self.name = name
        self.left = left
        self.right = right
        self.ascending = ascending

    def __str__(self):
        return '<%s:%s range %d %s %d>'%(self.name,self.ultimate_base.name,self.left,'to' if self.ascending else 'downto',self.right)

    @property
    def low(self):
        return min(self.left,self.right)

    @property
    def high(self):
        return max(self.left,self.right)

    @property
    def length(self):
        return max(0,
            (self.right - self.left + 1)
            if self.ascending else
            (self.left - self.right + 1)
        )

    def constrain(self,left,right,ascending=None,name=None):
        if self.left is not None:
            assert self.low <= min(left,right) and max(left,right)<=self.high
        return VHDLSubInteger(self,name,left,right,ascending)

    def equivalent_to(self,other):
        return isinstance(other,VHDLInteger) and self.left==other.left and self.right==other.right and self.ascending==other.ascending

    def castable_to(self,other):
        # TODO: check if these are the correct requirements (does ascending need to match?)
        return isinstance(other,VHDLInteger) and self.left>=other.left and self.right<=other.right and self.ascending==other.ascending

class VHDLSubInteger(VHDLSubtype,VHDLInteger):
    pass

class VHDLReal(VHDLScalar):
    pass

class VHDLEnumerated(VHDLScalar):
    _identity = ('name','values')
    def __init__(self,name,values=()):
        self.name = name
        self.values = tuple(values)

    def __str__(self):
        return '<%s:%s(%s)>'%(self.name,self.ultimate_base.name,', '.join(str(v) for v in self.values))

    @property
    def left(self):
        return self.values[0]

    @property
    def right(self):
        return self.values[-1]

    @property
    def length(self):
        return len(self.values)

    def equivalent_to(self,other):
        return isinstance(other,VHDLEnumerated) and self.values==other.values

    def castable_to(self,other):
        return self.compatible_with(other)

class VHDLSubEnum(VHDLSubtype,VHDLEnumerated):
    pass

class VHDLComposite(VHDLType):
    pass

class VHDLArray(VHDLComposite):
    _identity = ('name','valuetype','indextypes')
    def __init__(self,name,valuetype,*indextypes):
        self.name = name
        self.valuetype = valuetype
        self.indextypes = indextypes

    def __str__(self):
        return '<%s:%s array (%s) of %s>'%(self.name,self.ultimate_base.name,', '.join(str(v) for v in self.indextypes),self.valuetype)

    def constrain(self,name=None,*indextypes):
        return VHDLSubArray(self,name,*indextypes)

    def __getitem__(self,item):
        if isinstance(item,slice):
            return self[(item,)]
        if not len(self.indextypes) == len(item):
            raise IndexError('You must specify a constraint for every index in an array specification')
        return self.constrain(None,*[
            t.constrain(
                s.start if s.start is not None else t.left,
                s.stop if s.stop is not None else t.right,
            )
            for t,s in zip(self.indextypes,item)
        ])

    def equivalent_to(self,other):
        return (
            isinstance(other,VHDLArray)
            and self.valuetype.equivalent_to(other.valuetype)
            and len(self.indextypes) == len(other.indextypes)
            and all(s.equivalent_to(o) for s,o in zip(self.indextypes,other.indextypes))
        )

    def castable_to(self, other):
        if not isinstance(other, VHDLArray) or not self.valuetype.castable_to(other.valuetype):
            return False
        if len(self.indextypes) != len(other.indextypes):
            return False
        return all(s.castable_to(o) for s,o in zip(self.indextypes, other.indextypes))

    def compatible_with(self,other):
        if not isinstance(other,VHDLArray) or not self.ultimate_base == other.ultimate_base:
            return False
        return self.unconstrained or other.unconstrained or tuple(i.length for i in self.indextypes) == tuple(i.length for i in other.indextypes)

    unconstrained = True

class VHDLSubArray(VHDLSubtype,VHDLArray):
    unconstrained = False

    def __init__(self,base,name,*indextypes):
        if not len(base.indextypes) == len(indextypes) or not all(
            s.compatible_with(b) for s,b in zip(
                indextypes, base.indextypes
            )
        ):
            raise TypeError('The index of an array subtype must be compatible with the index of the base array')
        self.base = base
        self.name = name
        self.indextypes = indextypes

    def constrain(self,name=None,*indextypes):
        raise TypeError('Cannot constrain already constrained arrays')

    @property
    def valuetype(self):
        return self.base.valuetype

class VHDLRecord(VHDLComposite):
    pass


# - Standard types
bit = VHDLEnumerated('bit',(0,1))
boolean = VHDLEnumerated('boolean',(False,True))
character = VHDLEnumerated('character',tuple(chr(k) for k in range(32,128))) # TODO: verify correct set

integer = VHDLInteger('integer',-2**31+1,2**31-1)
natural = VHDLInteger('natural',0,2**31-1)
positive = VHDLInteger('positive',1,2**31-1)

severity_level = VHDLEnumerated('severity_level','note warning error failure'.split())

bit_vector = VHDLArray('bit_vector',bit,natural)
string = VHDLArray('string',character,natural)



# - std_logic_1164
std_ulogic = VHDLEnumerated('std_ulogic',tuple('UX01ZWLH-'))
std_logic = VHDLEnumerated('std_logic',tuple('UX01ZWLH-')) # TODO: implement resolution behaviour?

std_ulogic_vector = VHDLArray('std_ulogic_vector',std_ulogic,natural)
std_logic_vector = VHDLArray('std_logic_vector',std_logic,natural)
signed = VHDLArray('signed',std_logic,natural)
unsigned = VHDLArray('unsigned',std_logic,natural)


# -------------------
# VHDL AST
# -------------------

class VHDLTypeMapping(MigenExpressionType):
    """ Represents a Migen expression type including information on how it is mapped to VHDL.

    This includes the specification of both the Migen type and the VHDL type,
    as well as an implicit description of the mapping between the two.

    The semantics are still Migen semantics, governed by the Migen type.
    """
    def __init__(self,migen,vhdl):
        assert isinstance(migen, MigenExpressionType)
        assert isinstance(vhdl, VHDLType)
        self.migen = migen
        self.vhdl = vhdl


class AbstractTypedExpression(_Value):
    """ AST node with explicit type information.
    """
    def __init__(self,type):
        assert isinstance(type, MigenExpressionType)
        self.type = type

class MigenExpression(AbstractTypedExpression):
    """ A proxy for any Migen expression, with added explicit type information.
    """
    def __init__(self,expr,type):
        assert isinstance(expr,_Value) and not isinstance(expr,AbstractTypedExpression)
        super().__init__(type)
        self.expr = expr

class VHDLSignal(AbstractTypedExpression):
    def __init__(self,name,type):
        assert isinstance(type,VHDLTypeMapping)
        super().__init__(type)
        self.name = name

class Port(VHDLSignal):
    def __init__(self,name,dir,type):
        assert dir in ['in', 'out', 'inout', 'buffer']
        super().__init__(name,type)
        self.dir = dir

class Entity:
    def __init__(self,name,ports=[]):
        assert all(isinstance(p,Port) for p in ports)
        self.name = name
        self.ports = ports

class Component:
    def __init__(self,name,entity,**kw):
        assert not kw
        assert isinstance(entity,Entity)
        self.name = name
        self.entity = entity

class ComponentInstance(_Statement):
    """An instantiation of a component.

    This is a concurrent statement.
    """
    def __init__(self,name,component,portmap={},*kw):
        assert not kw
        assert isinstance(component,Component)
        self.name = name
        self.component = component
        self.portmap = portmap


class Architecture:
    def __init__(self,entity,**kw):
        name = kw.pop('name','Migen')
        signals = kw.pop('signals',dict())
        instances = kw.pop('instances',dict())
        assert not kw
        assert isinstance(entity,Entity)
        assert all(isinstance(s,VHDLSignal) for s in signals.values())
        assert all(isinstance(i,ComponentInstance) for i in instances.values())
        self.name = name
        self.entity = entity
        self.signals = signals
        self.instances = instances

class TypeConversion(AbstractTypedExpression):
    """ Changes the representation of a value, while keeping it's semantics unharmed.
    """

class TypeCast(AbstractTypedExpression):
    """ Changes the semantics of a value, while keeping it's representation unharmed.
    """

# ---------------------------
#
# ---------------------------

class VHDLNodeContext:
    concurrent = False  # are we in a concurrent situation or not?
    allowed_type = None # for expressions, either a single type or a tuple of types indicating the VHDL types that are valid at this point
    ns = None           # the currently active namespace
    def __init__(self,**kw):
        self.ns = {}
        self.update(**kw)

    def update(self,**kw):
        for k,v in kw.items():
            setattr(self,k,v)

class ToVHDLConverter(ContextualTransformer):
    """ Converts a Migen AST into an AST suitable for VHDL export

    In the process, identifies types of expressions and a
    suitable VHDL representation, and inserts type conversions where needed.
    """

    def type_wrap(self,node,ctxt,type):
        if ctxt.wrapped:
            return node
        return MigenExpression(node,type)

    @visitor_for(Constant)
    def visit_Constant(self, node, ctxt):
        return self.type_wrap(node,ctxt,(Signed if node.signed else Unsigned)(node.nbits))

    def visit_Signal(self, node):
        name = self.ns.get_name(node)
        return name, self.signal_types[node]

    def visit_ClockSignal(self, node):
        return self.visit_Signal(node)

    def visit_ResetSignal(self, node):
        return self.visit_Signal(node)

    @visitor_for(_Operator)
    def visit_Operator(self, node, ctxt):
        verilog_bits, verilog_signed = value_bits_sign(node)
        op = Verilog2VHDL_operator_map[node.op]  # TODO: op=='m'
        if op in {'and','or','nand','nor','xor','xnor'}:
            # logical operators
            left,right = node.operands
            lex,type = self.visit(left)
            rex = self.visit_as_type(right,type)
            return '('+lex + ' ' + op + ' ' + rex+')', type
        elif op in {'<','<=','=','/=','>','>='}:
            # relational operators
            left,right = node.operands
            lex,type = self.visit(left)
            rex = self.visit_as_type(right,type)
            return '('+lex + op + rex+')', boolean
        elif op in {'sll','srl','sla','sra','rol','ror'}:
            # shift operators
            left,right = node.operands
            lex,type = self.visit(left)
            rex = self.visit_as_type(right,integer)
            return '('+lex + ' ' + op + ' ' + rex+')', type
        elif op in {'+','-'} and len(node.operands)==2:
            # addition operators
            left,right = node.operands
            if False:
                # VHDL semantics
                lex,type = self.visit(left)
                rex = self.visit_as_type(right,type)
            else:
                # emulate Verilog semantics in VHDL
                type = (signed if verilog_signed else unsigned)[verilog_bits-1:0]
                lex = self.visit_as_type(left,type)
                rex = self.visit_as_type(right,type)
            return '('+lex + op + rex+')', type
        elif op in {'&'}:
            # concatenation operator (same precedence as addition operators)
            raise NotImplementedError(type(self).__name__ + '.visit_Operator: operator '+op)
        elif op in {'+','-'} and len(node.operands)==1:
            # unary operators
            right, = node.operands
            ex,type = self.visit(right)
            return '(' + op + ex+')', type
        elif op in {'*','/','mod','rem'}:
            # multiplying operators
            raise NotImplementedError(type(self).__name__ + '.visit_Operator: operator '+op)
        elif op in {'not'}:
            right, = node.operands
            ex,type = self.visit(right)
            return '(' + op + ' ' + ex+')', type
        elif op in {'**','abs'}:
            # misc operators
            raise NotImplementedError(type(self).__name__ + '.visit_Operator: operator '+op)
        else:
            raise TypeError('Unknown operator "%s" with %d operands'%(node.op,len(node.operands)))

    def visit_Slice(self, node):
        expr,type = self.visit(node.value)
        if not isinstance(type,VHDLArray):
            raise TypeError('Cannot slice value of non-array type %s'%type)
        idx, = type.indextypes
        if node.stop - node.start == 1:
            # this is not a slice, but an indexing operation!
            return expr + '(' + str(node.start) + ')', type.valuetype
        return expr + '(' + str(node.stop-1) + ' downto ' + str(node.start) + ')', type.ultimate_base[node.stop-1:node.start]

    def visit_Cat(self, node):
        pieces = []
        nbits = 0
        for o in node.l:
            expr,type = self.visit(o)
            if not isinstance(type,VHDLArray):
                pieces.append(self._convert_type(std_logic,expr,type))
 #               pieces.append(expr)
                nbits += 1
            else:
                l = type.indextypes[0].length
                pieces.append(self._convert_type(unsigned[l-1:0],expr,type))
#                pieces.append(expr)
                nbits += l
        expr = "unsigned'(" + '&'.join(reversed(pieces)) + ')';
        return expr, unsigned[nbits-1:0]

    def visit_Replicate(self, node):
        raise NotImplementedError(type(self).__name__+'.visit_Replicate')

    def visit_Assign(self, node):
        return self._cannot_visit(node)

    def visit_If(self, node):
        return self._cannot_visit(node)

    def visit_Case(self, node):
        return self._cannot_visit(node)

    def visit_Fragment(self, node):
        return self._cannot_visit(node)

    def visit_statements(self, node):
        return self._cannot_visit(node)

    def visit_clock_domains(self, node):
        return self._cannot_visit(node)

    def visit_ArrayProxy(self, node):
        raise NotImplementedError(type(self).__name__+'.visit_ArrayProxy')
        return _ArrayProxy([self.visit(choice) for choice in node.choices],
            self.visit(node.key))

    def visit_unknown(self, node):
        return self._cannot_visit(node)

    def _cannot_visit(self, node):
        raise TypeError('Node of type "%s" cannot be written as a VHDL expression'%type(node).__name__)


Verilog2VHDL_operator_map = {v[0]:v[-1] for v in (v.split(':') for v in '''
 &:and |:or ^:xor
 < <= ==:= !=:/= > >=
 <<:sll >>:srl <<<:sla >>>:sra
 + -
 *
 ~:not
'''.split())}

def conv(format):
    def convert(type,expr,orig=None):
        return format.format(
            x=expr,
            l=type.indextypes[0].length if isinstance(type,VHDLArray) else None,
        )
    return convert
standard_type_conversions = {
    (integer,std_logic):conv('to_integer(to_unsigned({x},1))'),   # '1', 'H' -> 1, else 0
    (integer,signed):conv('to_integer({x})'),   # one-to-one
    (integer,unsigned):conv('to_integer({x})'), # one-to-one
    (signed,integer):conv('to_signed({x},{l})'), # ?
    (unsigned,std_logic):conv('to_unsigned({x},{l})'),   # '1','H' -> 1, else -> 0
    (unsigned,integer):conv('to_unsigned({x},{l})'),   # ?
    (std_logic,boolean):conv("to_std_ulogic({x})"), # true -> '1', false -> '0'
    (std_logic,integer):conv("to_std_ulogic({x})"), # true -> 1, false -> 0
    (std_logic,unsigned):conv("to_std_ulogic(to_integer({x}) /= 0)"), # 0 -> '0', else -> '1'
    (boolean,std_logic):conv("({x} = '1')"),
    (boolean,integer):conv('({x} /= 0)'),
    (boolean,unsigned):conv("(to_integer({x}) /= 0)"), # 0 -> '0', else -> '1'
}


class VHDLExprPrinter(NodeTransformer):
    def __init__(self,ns,signal_types,conversion_functions = {}):
        self.ns = ns
        self.signal_types = signal_types
        tmp = dict(standard_type_conversions)
        tmp.update(conversion_functions)
        self.conversion_functions = tmp

    def _convert_type(self, type, expr, orig_type):
        if orig_type.compatible_with(type):
            return expr
        if orig_type.castable_to(type) and type.name is not None:
            return type.name + '(' + expr + ')'
        converter = self.conversion_functions.get((type.ultimate_base,orig_type.ultimate_base),None)
        if converter is not None:
            return converter(type,expr,orig_type)
        if not isinstance(orig_type,VHDLArray) or not isinstance(type,VHDLArray):
            raise TypeError("Don't know how to convert type %s to %s"%(orig_type,type))
        # simply assume the presence of a resize function
        expr = 'resize('+expr+','+str(type.indextypes[0].length)+')'
        if not orig_type.valuetype.compatible_with(type.valuetype):
            assert orig_type.castable_to(type.valuetype)
            assert type.name is not None
            expr = type.name+'('+expr+')'
        return expr

    def visit_as_type(self, node, type):
        # check for type conversions
        expr, orig_type = self.visit(node)
        return self._convert_type(type,expr,orig_type)

    def visit_Constant(self, node):
        return str(node.value),integer.constrain(node.value,node.value)

    def visit_Signal(self, node):
        name = self.ns.get_name(node)
        return name, self.signal_types[node]

    def visit_ClockSignal(self, node):
        return self.visit_Signal(node)

    def visit_ResetSignal(self, node):
        return self.visit_Signal(node)

    def visit_Operator(self, node):
        verilog_bits, verilog_signed = value_bits_sign(node)
        op = Verilog2VHDL_operator_map[node.op]  # TODO: op=='m'
        if op in {'and','or','nand','nor','xor','xnor'}:
            # logical operators
            left,right = node.operands
            lex,type = self.visit(left)
            rex = self.visit_as_type(right,type)
            return '('+lex + ' ' + op + ' ' + rex+')', type
        elif op in {'<','<=','=','/=','>','>='}:
            # relational operators
            left,right = node.operands
            lex,type = self.visit(left)
            rex = self.visit_as_type(right,type)
            return '('+lex + op + rex+')', boolean
        elif op in {'sll','srl','sla','sra','rol','ror'}:
            # shift operators
            left,right = node.operands
            lex,type = self.visit(left)
            rex = self.visit_as_type(right,integer)
            return '('+lex + ' ' + op + ' ' + rex+')', type
        elif op in {'+','-'} and len(node.operands)==2:
            # addition operators
            left,right = node.operands
            if False:
                # VHDL semantics
                lex,type = self.visit(left)
                rex = self.visit_as_type(right,type)
            else:
                # emulate Verilog semantics in VHDL
                type = (signed if verilog_signed else unsigned)[verilog_bits-1:0]
                lex = self.visit_as_type(left,type)
                rex = self.visit_as_type(right,type)
            return '('+lex + op + rex+')', type
        elif op in {'&'}:
            # concatenation operator (same precedence as addition operators)
            raise NotImplementedError(type(self).__name__ + '.visit_Operator: operator '+op)
        elif op in {'+','-'} and len(node.operands)==1:
            # unary operators
            right, = node.operands
            ex,type = self.visit(right)
            return '(' + op + ex+')', type
        elif op in {'*','/','mod','rem'}:
            # multiplying operators
            raise NotImplementedError(type(self).__name__ + '.visit_Operator: operator '+op)
        elif op in {'not'}:
            right, = node.operands
            ex,type = self.visit(right)
            return '(' + op + ' ' + ex+')', type
        elif op in {'**','abs'}:
            # misc operators
            raise NotImplementedError(type(self).__name__ + '.visit_Operator: operator '+op)
        else:
            raise TypeError('Unknown operator "%s" with %d operands'%(node.op,len(node.operands)))

    def visit_Slice(self, node):
        expr,type = self.visit(node.value)
        if not isinstance(type,VHDLArray):
            raise TypeError('Cannot slice value of non-array type %s'%type)
        idx, = type.indextypes
        if node.stop - node.start == 1:
            # this is not a slice, but an indexing operation!
            return expr + '(' + str(node.start) + ')', type.valuetype
        return expr + '(' + str(node.stop-1) + ' downto ' + str(node.start) + ')', type.ultimate_base[node.stop-1:node.start]

    def visit_Cat(self, node):
        pieces = []
        nbits = 0
        for o in node.l:
            expr,type = self.visit(o)
            if not isinstance(type,VHDLArray):
                pieces.append(self._convert_type(std_logic,expr,type))
 #               pieces.append(expr)
                nbits += 1
            else:
                l = type.indextypes[0].length
                pieces.append(self._convert_type(unsigned[l-1:0],expr,type))
#                pieces.append(expr)
                nbits += l
        expr = "unsigned'(" + '&'.join(reversed(pieces)) + ')';
        return expr, unsigned[nbits-1:0]

    def visit_Replicate(self, node):
        raise NotImplementedError(type(self).__name__+'.visit_Replicate')

    def visit_Assign(self, node):
        return self._cannot_visit(node)

    def visit_If(self, node):
        return self._cannot_visit(node)

    def visit_Case(self, node):
        return self._cannot_visit(node)

    def visit_Fragment(self, node):
        return self._cannot_visit(node)

    def visit_statements(self, node):
        return self._cannot_visit(node)

    def visit_clock_domains(self, node):
        return self._cannot_visit(node)

    def visit_ArrayProxy(self, node):
        raise NotImplementedError(type(self).__name__+'.visit_ArrayProxy')
        return _ArrayProxy([self.visit(choice) for choice in node.choices],
            self.visit(node.key))

    def visit_unknown(self, node):
        return self._cannot_visit(node)

    def _cannot_visit(self, node):
        raise TypeError('Node of type "%s" cannot be written as a VHDL expression'%type(node).__name__)

class _MapProxy(object):
    def __init__(self,getter):
        self._getter = getter

    def __getitem__(self,item):
        return self._getter(item)


literal_printers = {
    integer: lambda v, l: str(None),
    unsigned: lambda v, l: '"' + bin(v)[2:].rjust(l, '0') + '"',
    signed: lambda v, l: '"' + bin(v & ~-(1 << l))[2:].rjust(l, '0') + '"',
    std_logic: lambda v, l: "'1'" if v else "'0'",
}


class Converter:
    def typeof(self,sig):
        """ Calculate the VHDL type of a given expression/signal/variable. """
        if len(sig)==1:
            return std_logic
        base = signed if sig.signed else unsigned
        return base[len(sig)-1:0]

    def _printliteral(self, node, type):
        assert isinstance(node, Constant), "Choices in case statements must be constants"
        cp = literal_printers.get(type.ultimate_base, None)
        if cp is None:
            raise TypeError("Don't know how to print a constant of type %s"%type)
        l = type.indextypes[0].length if isinstance(type, VHDLArray) else None
        return cp(node.value, l)

    def _printsig(self, ns, s, dir, initialise=False):
        n = ns.get_name(s) + ': ' + dir
        if len(s) > 1:
            if s.signed:
                n += " signed"
            else:
                n += " unsigned"
            n += "(" + str(len(s) - 1) + " downto 0)";
        else:
            n += " std_logic"
        if initialise:
            n += ' := ' + self._printliteral(s.reset, self.typeof(s))
        return n

    def _printexpr(self, ns, node, type=None):
        printer = VHDLExprPrinter(ns,_MapProxy(self.typeof))
        if type is not None:
            return printer.visit_as_type(node,type)
        return printer.visit(node)[0]

    def _printnode(self, ns, level, node):
        if isinstance(node, _Assign):
            assignment = " <= "
            assert isinstance(node.l,(Signal,_Slice))
            left,leftt = VHDLExprPrinter(ns,_MapProxy(self.typeof)).visit(node.l)
#            lt = VHDLExprPrinter(ns,_MapProxy(self.typeof)).visit(node.l)[1]
#            rt = VHDLExprPrinter(ns,_MapProxy(self.typeof)).visit(node.r)[1]
#            print('Assign to ',ns.get_name(node.l),left,lt,rt)
            return "\t" * level + left + assignment + self._printexpr(ns, node.r, type=leftt) + ";\n"
        elif isinstance(node, collections.Iterable):
            return "".join(self._printnode(ns, level, n) for n in node)
        elif isinstance(node, If):
            r = "\t" * level + "if " + self._printexpr(ns, node.cond, boolean) + " then\n"
            r += self._printnode(ns, level + 1, node.t)
            if node.f:
                r += "\t" * level + "else\n"
                r += self._printnode(ns, level + 1, node.f)
            r += "\t" * level + "end if;\n"
            return r
        elif isinstance(node, Case):
            if node.cases:
                printer = VHDLExprPrinter(ns, _MapProxy(self.typeof))
                test,testt = printer.visit(node.test)

                r = "\t" * level + "case " + test + " is \n"
                css = [(k, v) for k, v in node.cases.items() if isinstance(k, Constant)]
                css = sorted(css, key=lambda x: x[0].value)
                for choice, statements in css:
                    r += "\t" * (level + 1) + "when " + self._printliteral(choice, testt) + " =>\n"
                    r += self._printnode(ns, level + 2, statements)
                if "default" in node.cases:
                    r += "\t" * (level + 1) + "when others => \n"
                    r += self._printnode(ns, level + 2, node.cases["default"])
                r += "\t" * level + "end case;\n"
                return r
            else:
                return ""
        else:
            raise TypeError("Node of unrecognized type: " + str(type(node)))

    def _printuse(self, extra=[]):
        r = """
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package migen_helpers is
function to_std_ulogic(v: boolean) return std_ulogic;
function to_std_ulogic(v: integer) return std_ulogic;
function to_unsigned(v: std_ulogic; length: natural) return unsigned;
end migen_helpers;

package body migen_helpers is
function to_std_ulogic(v: boolean) return std_ulogic is
begin
  if v then
    return '1';
  else
    return '0';
  end if;
end to_std_ulogic;

function to_std_ulogic(v: integer) return std_ulogic is
begin
  if v /= 0 then
    return '1';
  else
    return '0';
  end if;
end to_std_ulogic;

function to_unsigned(v: std_ulogic; length:natural) return unsigned is
begin
  if (v = '1') or (v = 'H') then
    return resize(unsigned'("1"),length);
  else
    return resize(unsigned'("0"),length);
  end if;
end to_unsigned;
end migen_helpers;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.migen_helpers.all;
"""
        for u in extra:
            r += 'use '+u+';\n'
        return r

    def _printentitydecl(self, f, ios, name, ns,
                         reg_initialization):
        sigs = list_signals(f) | list_special_ios(f, True, True, True)
        special_outs = list_special_ios(f, False, True, True)
        inouts = list_special_ios(f, False, False, True)
        targets = list_targets(f) | special_outs
        r = """
entity {name} is
\tport(
""".format(name=name)
        r += ';\n'.join(
            '\t' * 2 + self._printsig(ns, sig, 'inout' if sig in inouts else 'buffer' if sig in targets else 'in', initialise=True)
            for sig in sorted(ios, key=lambda x: x.duid)
        )
        r += """\n);
end {name};
    """.format(name=name)
        return r

    def _printarchitectureheader(self, f, ios, name, ns,
                                 reg_initialization, extra=""):
        sigs = list_signals(f) | list_special_ios(f, True, True, True)
        special_outs = list_special_ios(f, False, True, True)
        inouts = list_special_ios(f, False, False, True)
        targets = list_targets(f) | special_outs
        r = """
    architecture Migen of {name} is
    """.format(name=name)
        r += '\n'.join(
            ' ' * 4 + 'signal ' + self._printsig(ns, sig, '', initialise=True) + ';'
            for sig in sorted(sigs - ios, key=lambda x: x.duid)
        ) + '\n'
        r += extra
        r += "begin\n"
        return r

    def _printsync(self, f, name, ns):
        r = ""
        for k, v in sorted(f.sync.items(), key=itemgetter(0)):
            clk = ns.get_name(f.clock_domains[k].clk)
            r += name+'_'+k+": process ({clk})\nbegin\nif rising_edge({clk}) then\n".format(clk=clk)
            r += self._printnode(ns, 1, v)
            r += "end if;end process;\n\n"
        return r

    def _printcomb(self, f, ns):
        if not f.comb:
            return '\n'

        r = ""
        groups = group_by_targets(f.comb)

        def extract_assignment(target,node,type,reset, precondition=None):
            if isinstance(node,_Assign):
                if node.l is target:
                    return self._printexpr(ns,node.r,type)
                return None
            if isinstance(node,If):
                condition = self._printexpr(ns, node.cond, boolean)
                if precondition:
                    condition = '(' + precondition + ' and ' + condition + ')'
                return (
                    extract_assignment(target,node.t,type,reset, condition)
                    +  " when " + condition + " else "
                    + (extract_assignment(target,node.f,type,reset) if node.f else reset)
                )
            if isinstance(node,(list,tuple)):
                values = list(filter(None,[extract_assignment(target,n,type,reset) for n in node]))
                if len(values) > 1:
                    raise TypeError('More than one assignment to '+str(target))
                if not values:
                    return reset
                return values[0]
            raise TypeError('Combinatorial statements may only contain _Assign or If:\n'+format_tree(node,prefix='> '))

        for n, (target,statements) in enumerate(groups):
            for t in target:
                type = self.typeof(t)
                reset = self._printexpr(ns,t.reset, type)
                r += "\t" + ns.get_name(t) + " <= " + extract_assignment(t,statements,type,reset) + ";\n"
        r += "\n"
        return r

    def _printspecials(self, overrides, specials, ns, add_data_file):
        use = []
        decl = ""
        body = ""
        for special in sorted(specials, key=lambda x: x.duid):
            pr = call_special_classmethod(overrides, special, "emit_vhdl", self, ns, add_data_file)
            if pr is None:
                raise NotImplementedError("Special " + str(special) + " failed to implement emit_vhdl")
            use.extend(pr.get('use',[]))
            decl += pr.get('decl','')
            body += pr.get('body','')
        return dict(use=use,decl=decl,body=body)

    def convert(self, f, ios=None, name="top",
        special_overrides=dict(),
        create_clock_domains=True,
        asic_syntax=False
    ):
        r = ConvOutput()
        if not isinstance(f, _Fragment):
            f = f.get_fragment()
        if ios is None:
            ios = set()
        r.fragment = f
        r.name = name

        for cd_name in sorted(list_clock_domains(f)):
            try:
                f.clock_domains[cd_name]
            except KeyError:
                if create_clock_domains:
                    cd = ClockDomain(cd_name)
                    f.clock_domains.append(cd)
                    ios |= {cd.clk, cd.rst}
                else:
                    raise KeyError("Unresolved clock domain: '"+cd_name+"'")

        r.ios = ios

        f = lower_complex_slices(f)
        insert_resets(f)
        f = lower_basics(f)
        fs, lowered_specials = lower_specials(special_overrides, f.specials)
        f += lower_basics(fs)

        r.lowered_fragment = f

        for io in sorted(ios, key=lambda x: x.duid):
            if io.name_override is None:
                io_name = io.backtrace[-1][0]
                if io_name:
                    io.name_override = io_name
        ns = build_namespace(list_signals(f) \
                             | list_special_ios(f, True, True, True) \
                             | ios, _reserved_keywords)
        ns.clock_domains = f.clock_domains
        r.ns = ns

        specials = self._printspecials(special_overrides, f.specials - lowered_specials, ns, r.add_data_file)


        src = "-- Machine-generated using Migen\n"
        src += self._printuse(extra=specials['use'])
        src += self._printentitydecl(f, ios, name, ns, reg_initialization=not asic_syntax)
        src += self._printarchitectureheader(f, ios, name, ns, reg_initialization=not asic_syntax, extra=specials['decl'])
        src += self._printcomb(f, ns)
        src += self._printsync(f, name, ns)
        src += specials['body']
        src += "end Migen;\n"
        r.set_main_source(src)

        return r

    def generate_testbench(self, code, clocks={'sys':10}):
        """ Genertes a testbench that does nothing but instantiate the DUT and run it's clocks.

        The testbench does not generate any reset signals.

        You must supply the result of a previous conversion.
        """
        from ..sim.core import TimeManager

        name = code.name
        ns = code.ns
        f = code.lowered_fragment
        ios = code.ios

        tbname = name + '_testbench'

        sigs = list_signals(f) | list_special_ios(f, True, True, True)
        special_outs = list_special_ios(f, False, True, True)
        inouts = list_special_ios(f, False, False, True)
        targets = list_targets(f) | special_outs

        time = TimeManager(clocks)

        sortedios = sorted(ios, key=lambda x: x.duid)
        src = """
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity {name} is
begin
end {name};

architecture Migen of {name} is

procedure clk_gen(
    signal clk : out std_logic;
    constant half_period : time;
--    signal run : in std_logic;
    constant first_edge_advance : time := 0 fs;
    constant initial_level: std_logic := '0'
) is
begin
  -- Check the arguments
  assert (half_period /= 0 fs) report "clk_gen: half_period is zero; time resolution to large for frequency" severity FAILURE;
  -- Initial phase shift
  clk <= initial_level;
  wait for half_period - first_edge_advance;
  if initial_level /= '0' then
    clk <= '0';
    wait for half_period;
  end if;
  -- Generate cycles
  loop
    -- Only high pulse if run is '1' or 'H'
--    if (run = '1') or (run = 'H') then
--      clk <= run;
--    end if;
    clk <= '1';
    wait for half_period;
    -- Low part of cycle
    clk <= '0';
    wait for half_period;
  end loop;
end procedure;

component {dut}
\tport({dutport});
end component;
{signaldecl}
begin
\tdut: {dut} port map ({portmap});
""".format(
            name=tbname,
            dut=name,
            dutport=';\n\t\t'.join(
                self._printsig(ns, sig, 'inout' if sig in inouts else 'buffer' if sig in targets else 'in', initialise=False)
                for sig in sortedios
            ),
            signaldecl=''.join(
                'signal ' + self._printsig(ns, sig, '', initialise=True) + ';\n'
                for sig in sortedios
            ),
            portmap=', '.join(ns.get_name(s) for s in sortedios),
        )

        for k in sorted(list_clock_domains(f)):
            clk = time.clocks[k]
            src += """clk_gen({name}_clk, {dt} ns, {advance} ns, {initial}); """.format(
                name=k,
                dt=clk.half_period,
                advance=clk.half_period - clk.time_before_trans,
                initial="'1'" if clk.high else "'0'",
            )

        src += 'end;\n'

        return src

def convert(f, ios=None, name="top", special_overrides={}, create_clock_domains=True, asic_syntax=False):
    return Converter().convert(f,ios,name,special_overrides,create_clock_domains,asic_syntax)
