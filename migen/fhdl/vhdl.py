from functools import partial
from operator import itemgetter
import abc, itertools
import collections
import abc

from migen.fhdl.structure import *
from migen.fhdl.structure import _Operator, _Slice, _Assign, _Fragment, _Value, _Statement, _ArrayProxy
from migen.fhdl.tools import *
from migen.fhdl.visit import NodeTransformer
from migen.fhdl.visit_generic import (
    NodeTransformer as NodeTranformerGeneric,
    visitor_for, recursor_for, context_for, combiner_for,
)
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
    def nbits(self):
        """ The number of bits required to completely represent the type.
        """

    @abc.abstractmethod
    def compatible_with(self,other):
        """ Tests if the set of values representable by the two types overlap.

        If the two sets do not overlap at all, then it is certainly an error to
        assume a variable to be in the intersection of the two, as required
        for type casting.
        """

    @abc.abstractmethod
    def contained_in(self,other):
        """ Tests if the set of values representable in self is contained in the set of other.

        In this case, casting is guaranteed to work.
        """

class MigenBool(MigenExpressionType):
    @property
    def nbits(self):
        return 1

    def compatible_with(self, other):
        if isinstance(other, MigenBool):
            return True
        return False

    def contained_in(self, other):
        if isinstance(other, MigenBool):
            return True
        return False


class MigenInteger(MigenExpressionType):
    def __init__(self,width,min=None,max=None):
        if min is None:
            min = self._min
        else:
            assert self._min <= min
        if max is None:
            max = self._max
        else:
            assert max <= self._max
        self._width = width
        self.min = min
        self.max = max

    @property
    def nbits(self):
        return self._width

    def compatible_with(self,other):
        if not isinstance(other,MigenInteger):
            return False
        return self.max>=other.min and self.min<=other.max

    def contained_in(self,other):
        if not isinstance(other,MigenInteger):
            return False
        return self.min >= other.min and self.max <= other.max

    @abc.abstractproperty
    def _min(self):
        """ Lowest possible value.

        The allowed range is [min .. max] with both ends inclusive.
        """

    @abc.abstractproperty
    def _max(self):
        """ Highest possible value.

        The allowed range is [min .. max] with both ends inclusive.
        """

class Unsigned(MigenInteger):
    @property
    def _min(self): return 0
    @property
    def _max(self): return (1<<self._width) - 1

class Signed(MigenInteger):
    @property
    def _min(self): return -1<<(self._width-1)
    @property
    def _max(self): return (1<<(self._width-1)) - 1


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
#        print(self,' castable_to ',other,'?')
        if not isinstance(other, VHDLArray) or not self.valuetype.castable_to(other.valuetype):
#            print(self.valuetype,' not castable_to ',isinstance(other,VHDLArray) and other.valuetype,'!')
            return False
        if len(self.indextypes) != len(other.indextypes):
#            print(self.indextypes,' not the same dimensionality as ',other.indextypes,'!')
            return False
#        print('|'.join(str(s.castable_to(o)) for s,o in zip(self.indextypes, other.indextypes)))
        return all(s.castable_to(o) and s.length == o.length for s,o in zip(self.indextypes, other.indextypes))

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
# explicitely typed AST
# -------------------

class AbstractTypedExpression(_Value):
    """ AST node with explicit type information.

    Futhermore, there is an optional field to hold information on how it is to be represented by the backend.
    As such, it has no meaning for the actual model, and should only be added when transforming the tree
    for a particular backend.
    """
    def __init__(self,type,repr=None):
        assert isinstance(type, MigenExpressionType)
        self.type = type
        self.repr = repr

class MigenExpression(AbstractTypedExpression):
    """ A proxy for any Migen expression, with added explicit type information.

    The annotated type is required to exactly match the type inferred from the
    semantics of the wrapped expression. If you want to change the semantics or
    restrict the type, inject a type-cast node.
    """
    def __init__(self,expr,type,repr=None):
        assert isinstance(expr,_Value) and not isinstance(expr,AbstractTypedExpression)
        AbstractTypedExpression.__init__(expr,type,repr)
        self.expr = expr

class VHDLSignal(AbstractTypedExpression):
    def __init__(self,name,type,repr=None):
        AbstractTypedExpression.__init__(type,repr)
        self.name = name

class Port(VHDLSignal):
    def __init__(self,name,dir,type,repr=None):
        assert dir in ['in', 'out', 'inout', 'buffer']
        VHDLSignal.__init__(name,type,repr)
        self.dir = dir


class Scope(abc.ABC):
    """ Mixin class for AST nodes that form a scope or namespace. """
    def get_object(self,name):
        ret = self._get_object(name) or self.parent_scope and self.parent_scope.get_object(name)
        if ret is None:
            raise KeyError('Name "%s" is undefined in scope "%s"'%(name,self))
        return ret

    @property
    def parent_scope(self):
        return ReservedKeywords

    @abc.abstractmethod
    def _get_object(self, name):
        pass

class DictScope(dict,Scope):
    def _get_object(self,name):
        return self[name]

ReservedKeyword = ()    # Singleton indicating that a name is a reserved keyword
ReservedKeywords = DictScope({
    k:ReservedKeyword for k in _reserved_keywords
})

class Entity(Scope):
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

class EntityBody(Scope):
    def __init__(self,entity,*,statements=[],architecture='Migen',signals={},instances={},namespace={}):
        """
        :param entity: The entity which is implemented in this body
        :param architecture: The architecture name
        :param signals: The local signals (ports are defined on `entity`)
        :param instances: Instantiations of components
        :param namespace: Mapping of all names to their meaning inside the body.
        :param statements: The (concurrent) statements (including processes) contained in the body.
        """
        assert isinstance(entity,Entity)
        assert all(isinstance(s,VHDLSignal) for s in signals.values())
        assert all(isinstance(i,ComponentInstance) for i in instances.values())
        self.architecture = architecture
        self.entity = entity
        self.statements = statements
        self.signals = signals
        self.instances = instances
        self.namespace = namespace

    @property
    def parent_scope(self):
        return self.entity

    def _get_object(self,name):
        return self.signals.get(name,None) or self.instances.get(name,None)

class TypeChange(AbstractTypedExpression):
    """ Replaces the representation and semantics of a type while preserving the
    represented value.

    Note that this might involve some logic in the underlying represenation or might
    simply be a type cast for the underlying representation.
    """
    def __init__(self,expr,*,type=None,repr=None):
        assert isinstance(expr,AbstractTypedExpression)
        super().__init__(
            type=type if type is not None else expr.type,
            repr=repr if repr is not None else expr.repr,
        )
        self.expr = expr

    @classmethod
    def if_needed(cls, expr, *, type=None, repr=None):
        ret = cls(expr, type=type, repr=repr)
        if ret.type == expr.type and ret.repr == expr.repr:
            return expr
        return ret

class TestIfNonzero(TypeChange):
    """ Explicit test for nonzero integer values, returning a boolean result.

    Note that migen's comparison operators return integers (either 0 or 1).
    If statements test for non-zero integers, while assigning to 1-bit registers
    truncates to the LSB.
    """
    def __init__(self,expr,*,repr=None):
        super().__init__(
            expr,
            type=MigenBool(),
            repr=repr
        )

class SelectedAssignment(_Assign):
    """ Essentially a case statement with a built-in assignment to a single target.

    In VHDL, assignments, processes and component instantiations are the only
    concurrent statements. Particularly, no case statements unless wrapped inside a process.
    However, selected assignments are specially designed assignments for this purpose.
    """
    def __init__(self,*args,**kw):
        raise NotImplementedError('SelectedAssignment')

class ConditionalAssignment(_Assign):
    """ Essentially if-then-else with a built-in assignment to a single target.

    In VHDL, assignments, processes and component instantiations are the only
    concurrent statements. Particularly, no if statements unless wrapped inside a process.
    However, conditional assignments are specially designed assignments for this purpose.
    """
    def __init__(self,l,default,condition_value_pairs):
        assert all(len(v)==2 and all(isinstance(vv,_Value) for vv in v) for v in condition_value_pairs)
        self.l = l
        self.defaul = default
        self.condition_value_pairs = condition_value_pairs

# ---- NodeTransformer for extended hierarchy

def visitor_for_wrapped(*wrapped_node_types,needs_original_node=True):

    """ Decorator to register a method to handle nodes wrapped within :py:class:`MigenExpression` nodes.

    Any :py:class:`MigenExpression` node whose `expr` attribute matches any of the node classes given
    as argument to this decorator will be handled by the decorated function.
    The decorated function receives either the single argument `node`, or both `orig` and `node`,
    where both are the wrapping nodes of type :py:class:`MigenExpression`. `orig` is the node before nested
    nodes were transformed, `node` is after nested transforms are applied.
    `orig` may be requested by setting the keyword argument `needs_original_node=True` on the decorator.
    """
    return NodeTranformerGeneric.registering_decorator(
        '_wrapped_node_visitors', wrapped_node_types,
        _needs_original_node=needs_original_node,
    )

def combiner_for_wrapped(*wrapped_node_types):

    """ Decorator to register a method to handle nodes wrapped within :py:class:`MigenExpression` nodes.

    Any :py:class:`MigenExpression` node whose `expr` attribute matches any of the node classes given
    as argument to this decorator will be handled by the decorated function.
    """
    return NodeTranformerGeneric.registering_decorator(
        '_wrapped_node_combiners', wrapped_node_types,
    )

class VHDLNodeTransformer(NodeTranformerGeneric):
    """ Extends the NodeTransformer to handler VHDL nodes properly.
    """
    @recursor_for(MigenExpression)
    def recurse_Annotated(self,node):
        return self.combine(node,self.visit(node.expr),node.type)

    @combiner_for(MigenExpression)
    def combine_Annotated(self, orig, expr, type):
        """ An annotated node. Delegates to the transform's registry based on the type of the wrapped expression."""
        handler = type(self).find_handler(
            '_wrapped_node_combiners',
            orig.expr,
            type(self).combine_unknown_wrapped_node
        )
        handler(self, orig, expr, type)


    def combine_unknown_wrapped_node(self, orig, expr, type):
        return type(orig)(expr,type)

    @visitor_for(MigenExpression, needs_original_node=True)
    def visit_Annotated(self, orig, node):
        """ An annotated node. Delegates to the transform's registry based on the type of the wrapped expression."""
        handler = type(self).find_handler(
            '_wrapped_node_visitors',
            node.expr,
            type(self).visit_unknown_wrapped_node
        )
        if getattr(handler, '_needs_original_node', None):
            handler(self, orig, node)
        else:
            handler(self, node)

    def visit_unknown_wrapped_node(self,node):
        return node

    @recursor_for(TypeChange)
    def recurse_TypeChange(self,node):
        return self.combine(node,self.visit(node.expr), type=node.type, repr=node.repr)
    @combiner_for(TestIfNonzero)
    def combine_TestIfNonzero(self,node,expr,*,type,repr):
        assert type == MigenBool()
        return type(node)(expr,repr=repr)

    @recursor_for(ConditionalAssignment)
    def recurse_ConditionalAssignment(self,node):
        return self.combine(
            node,
            self.visit(node.l),
            self.visit(node.default),
            self.visit([
                tuple(self.visit(v) for v in cv)
                for cv in node.condition_value_pairs
            ])
        )

    @recursor_for(SelectedAssignment)
    def recurse_SelectedAssignment(self,node):
        raise NotImplementedError('recurse_SelectedAssignment')

# ---------------------------
# Add explicit types to tree
# ---------------------------

class ExplicitTyper(VHDLNodeTransformer):
    """ Adds explicit type annotations to untyped Migen nodes.

    """
    def __init__(self,*,raise_on_type_mismatch=True):
        """
        raise_on_type_mismatch: If pre-existing type annotations do not match, raise an exception rather than just replacing
        the annotation.
        """
        self.raise_on_type_mismatch = raise_on_type_mismatch

    def type_wrap(self,node,type):
        """ Annotate node with type.
        """
        return MigenExpression(node,type)

    @visitor_for(AbstractTypedExpression)
    def visit_AlreadyTyped(self,node):
        """ Leave alone... this transformer is only concerned with nodes that are not inherently typed. """
        return node

    @visitor_for(MigenExpression,needs_original_node=True) # overrides visit_AlreadyTyped due to beeing more specific
    def visit_PreviousAnnotation(self,orig,node):
        ret = node.expr # we'll remove outer layer of annotation, which is the previous annotation
        assert isinstance(ret,MigenExpression)
        if self.raise_on_type_mismatch and not ret.type==orig.type:
            raise TypeError('Mismatching type annotations found for expression %s: %s != %s'%(ret.expr,ret.type,orig.type))
        return node

    @visitor_for(Constant)
    def visit_Constant(self, node):
        return self.type_wrap(node,(Signed if node.signed else Unsigned)(node.nbits,min=node.value,max=node.value))

    @visitor_for(Signal)
    def visit_Signal(self, node):
        return self.type_wrap(node,(Signed if node.signed else Unsigned)(node.nbits,min=node.min,max=node.max))

    @visitor_for(ClockSignal,ResetSignal)
    def visit_BoolSignal(self, node):
        return self.type_wrap(node,Unsigned(1,min=0,max=1))

    @visitor_for(_Operator)
    def visit_Operator(self, node):
        from .bitcontainer import operator_bits_sign
        obs = [(n.type.size,isinstance(n.type,Signed)) for n in node.operands]
        nbits, signed = operator_bits_sign(node.op, obs)
        typ = (Signed if signed else Unsigned)(nbits)
        return self.type_wrap(node,typ)

    @visitor_for(_Slice)
    def visit_Slice(self, node):
        typ = (Signed if isinstance(node.value.type,Signed) else Unsigned)(node.stop - node.start)
        return self.type_wrap(node,typ)

    @visitor_for(Cat)
    def visit_Cat(self, node):
        typ = Unsigned(sum(sv.type.nbits for sv in node.l))
        return self.type_wrap(node,typ)

    @visitor_for(Replicate)
    def visit_Replicate(self, node):
        typ = Unsigned(node.v.type.nbits*node.n)
        return self.type_wrap(node,typ)

    @visitor_for(_ArrayProxy)
    def visit_ArrayProxy(self, node):
        typ = (
            Signed if
            any(isinstance(n.type,Signed) for n in node.choices)
            else Unsigned
        )(max(n.type.nbits for n in node.choices))
        return self.type_wrap(node,typ)

    @visitor_for(_Value)
    def visit_other_Expression(self, node):
        # catch all for _Value, assume every expression is a subclass of _Value
        raise TypeError("Don't know how to generate type for expression node %s"%node)

    @visitor_for(If)
    def visit_If(self, node):
        # make implicit test explicit
        node.cond = TestIfNonzero(node.cond)
        return node

    def visit_unknown_node(self, node):
        # most likely not an expression, just ignore.
        return None

# ---------------------------
# Tree transforms for VHDL
# ---------------------------

if False:
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

class VHDLReprGenerator:
    def __init__(self,overrides={},*,single_bit=std_logic,unsigned=unsigned,signed=signed,boolean=boolean):
        self.overrides = overrides
        self.boolean = boolean
        self.single_bit = single_bit
        self.unsigned = unsigned
        self.signed = signed
        self.typer = ExplicitTyper(raise_on_type_mismatch=True)

    def VHDL_representation_of_Signal(self,signal):
        if not isinstance(signal,AbstractTypedExpression):
            signal = self.typer.visit(signal)
#        assert isinstance(signal,)
        migen_type = signal.type
        return migen_type, self.VHDL_representation_for(migen_type)

    def VHDL_representation_for(self, migen_type):
        """ Given a migen type, find a suitable VHDL representation.

        If argument already has a specified VHDL representation, returns that one.
        """
        assert isinstance(migen_type,MigenExpressionType)
        if isinstance(migen_type,MigenBool):
            return self.boolean
        if not isinstance(migen_type,MigenInteger):
            raise TypeError("Don't know how to map type %s to VHDL"%migen_type)
        if isinstance(migen_type, Unsigned) and migen_type.nbits == 1:
            # Verilog has no boolean type, so is this a 1-element array or a single wire?
            if self.single_bit is not None:
                return self.single_bit
        typ = self.signed if isinstance(migen_type, Signed) else self.unsigned
        return typ[migen_type.nbits-1:0]

natural_repr = VHDLReprGenerator(single_bit=std_logic,unsigned=unsigned,signed=signed,boolean=boolean)
only_numeric = VHDLReprGenerator(single_bit=None,unsigned=unsigned,signed=signed,boolean=unsigned[0:0])
all_slv = VHDLReprGenerator(single_bit=std_logic,unsigned=std_logic_vector,signed=std_logic_vector,boolean=std_logic)

class ToVHDLConverter(VHDLNodeTransformer):
    """ Converts a Migen AST into an AST suitable for VHDL export

    In this process, it assigns a suitable VHDL type to any expression.
    """

    def __init__(self,*,vhdl_repr=natural_repr,replaced_signals={}):
        # configuration (constants)
        self.vhdl_repr = vhdl_repr

        # global variables
        self.replaced_signals = replaced_signals

        # context variables
        self.entity = None  # the entity body we're currently building

    @context_for(EntityBody)
    def EntityBody_ctxt(self, node):
        return dict(
            entity = node,
        )

    def assign_VHDL_repr(self, node):
        if node.repr is None:
            node.repr = self.vhdl_repr.VHDL_representation_for(node.type)

    @visitor_for(AbstractTypedExpression)
    def visit_Typed(self, node):
        self.assign_VHDL_repr(node)
        return node

    @visitor_for(MigenExpression,needs_original_node=True)
    def visit_Annotated(self, orig, node):
        # nested nodes have already been processed, now generate an appropriate VHDL type for this node
        self.assign_VHDL_repr(node)
        # now delegate to visitors
        return super().visit_Annotated(orig,node)

    @visitor_for_wrapped(Signal)
    def wrapped_Signal(self, node):
        sig = node.expr
        repl = self.replaced_signals.get(sig,None)
        if repl is not None:
            # this Signal already has a replacement.
            # TODO: should we again check if the type in this wrapper conforms to the replacement signal?
            return repl
        # this Signal does not yet have a replacement.
        ret = VHDLSignal(
            name = self.ns.get_name(sig),
            type = node.type,
            repr = self.vhdl_repr.VHDL_representation_for(node.type),
        )
        # add to replacement map
        self.replaced_signals[sig] = ret
        # add signal to entity
        assert not ret.name in self.entity.ports
        assert not ret.name in self.entity.signals
        self.entity.signals[ret.name] = ret
        return ret

    @visitor_for_wrapped(_Operator)
    def visit_Operator(self, node):
        expr = node.expr
        repr = node.repr
        op = Verilog2VHDL_operator_map[expr.op]  # TODO: op=='m'
        if op in {
            'and', 'or', 'nand', 'nor', 'xor', 'xnor', # logical operators
            'not',                                     # unary logical operators
            '+', '-',                                  # addition operators (two operands)
            '<', '<=', '=', '/=', '>', '>=',           # relational operators
            '+', '-',                                  # unary operators (one operand)
        }:
            assert all(
                isinstance(v.type, MigenInteger)
                for v in expr.operands
            ) # only Migen semantics are implemented
            # VHDL and migen semantics match as long as the input types match
            # and are either signed or unsigned
            migen_repr = only_numeric.VHDL_representation_for(node.type)
            expr.operands = [
                TypeChange.if_needed(v,repr=migen_repr)
                for v in
                expr.operands
            ]
            return TypeChange.if_needed(node,repr=repr)
        elif op in {'sll', 'srl', 'sla', 'sra', 'rol', 'ror'}:
            # shift operators
            raise NotImplementedError(type(self).__name__ + '.visit_Operator: operator ' + op)
        elif op in {'&'}:
            # concatenation operator (same precedence as addition operators)
            raise NotImplementedError(type(self).__name__ + '.visit_Operator: operator ' + op)
        elif op in {'*', '/', 'mod', 'rem'}:
            # multiplying operators
            raise NotImplementedError(type(self).__name__ + '.visit_Operator: operator ' + op)
        elif op in {'**', 'abs'}:
            # misc operators
            raise NotImplementedError(type(self).__name__ + '.visit_Operator: operator ' + op)
        else:
            raise TypeError('Unknown operator "%s" with %d operands' % (node.op, len(node.operands)))

    @visitor_for(TestIfNonzero)
    def visit_TestIfNonzero(self,node):
        node.expr = TypeChange.if_needed(node.expr,repr=integer)
        repr = node.repr
        node.repr = boolean
        return TypeChange.if_needed(node,repr=repr)


class Converter:
    def __init__(self,*,io_repr=all_slv,create_clock_domains=True,special_overrides={}):
        self.io_repr = io_repr
        self.create_clock_domains = create_clock_domains
        self.special_overrides = special_overrides

    def convert(self,f,ios,name='top'):
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
                if self.create_clock_domains:
                    cd = ClockDomain(cd_name)
                    f.clock_domains.append(cd)
                    ios |= {cd.clk, cd.rst}
                else:
                    raise KeyError("Unresolved clock domain: '" + cd_name + "'")


        f = lower_complex_slices(f)
        insert_resets(f)
        f = lower_basics(f)
        fs, lowered_specials = lower_specials(self.special_overrides, f.specials)
        f += lower_basics(fs)

        # add explicit typing information
        f = ExplicitTyper().visit(f)

        r.lowered_fragment = f

        # generate outer structure
        ns = build_namespace(list_signals(f) \
                             | list_special_ios(f, True, True, True) \
                             | ios, _reserved_keywords)
        ns.clock_domains = f.clock_domains
        r.ns = ns

        ports = []
        replaced_signals = {}
        for io in sorted(ios, key=lambda x: x.duid):
            if io.name_override is None:
                io_name = io.backtrace[-1][0]
                if io_name:
                    io.name_override = io_name
            typ, rep = self.io_repr.VHDL_representation_for(io)
            p = Port(
                name = ns.get_name(io),
                dir = 'out' if io in outputs else 'in',
                type = typ,
                repr = rep,
            )
            replaced_signals[io] = p
            ports.append(p)
        r.ios = ios

        entity = Entity(name=name,ports=ports)
        entity_body = EntityBody(entity=entity,statements=[f])

        # convert body
        entity_body = ToVHDLConverter(
            vhdl_repr=self.vhdl_repr,
            replaced_signals=replaced_signals,
        ).visit(entity_body)
        r.replaced_signals = replaced_signals
        r.converted = entity_body

        # generate VHDL
        src = VHDLPrinter().visit([
            entity_body    # generates both the declaration and implementation of the entity
        ])
        r.set_main_source(
            src
        )

        return r

Verilog2VHDL_operator_map = {v[0]:v[-1] for v in (v.split(':') for v in '''
 &:and |:or ^:xor
 < <= ==:= !=:/= > >=
 <<:sll >>:srl <<<:sla >>>:sra
 + -
 *
 ~:not
'''.split())}


literal_printers = {
    integer: lambda v, l: str(v),
    unsigned: lambda v, l: '"' + bin(v)[2:].rjust(l, '0') + '"',
    signed: lambda v, l: '"' + bin(v & ~-(1 << l))[2:].rjust(l, '0') + '"',
    std_logic: lambda v, l: "'1'" if v else "'0'",
    boolean: lambda v, l: "true" if v else "false",
}

# ---------------------------------
# integer representation converters
#
# for all integers simultaneously representable in both representations
# the conversion should be one to one. Otherwise the result is undefined.
def conv(template):
    def convert(repr, expr, orig=None):
        return template.format(
            x=expr,
            l=repr.indextypes[0].length if isinstance(repr, VHDLArray) else None,
        )
    return convert
integer_repr_converters = {
    (integer,std_logic):conv('to_integer(to_unsigned({x},1))'),   # '1', 'H' -> 1, else 0
    (integer,signed):conv('to_integer({x})'),   # one-to-one
    (integer,unsigned):conv('to_integer({x})'), # one-to-one
    (signed,integer):conv('to_signed({x},{l})'), # ?
    (unsigned,std_logic):conv('to_unsigned({x},{l})'),   # '1','H' -> 1, else -> 0
    (unsigned,integer):conv('to_unsigned({x},{l})'),   # ?
    (std_logic,boolean):conv("to_std_ulogic({x})"), # '1','H' -> true, else -> false
    (integer,boolean):conv("to_integer(to_unsigned(to_std_ulogic({x}),1))"), # true -> '1', false -> '0'
    (std_logic,integer):conv("get_index(to_unsigned({x},1),0)"), # truncates
    (std_logic,unsigned):conv("get_index({x},0)"), # truncates
#    (boolean,std_logic):conv("({x} = '1')"),
    (boolean,integer):conv('({x} /= 0)'),
    (boolean,unsigned):conv("(to_integer({x}) /= 0)"), # 0 -> false, else -> true
}


class VHDLPrinter(VHDLNodeTransformer):
    def __init__(self):
        # configuration (constants)

        # global variables

        # context variables
        pass

    @visitor_for_wrapped(Constant)
    def visit_Constant(self, node):
        printer = literal_printers[node.repr]
        return printer(node.expr.value)

    @combiner_for(TypeChange)
    def combine_TypeChange(self, node, expr, *, type=None, repr=None):
        return expr

    @visitor_for(TypeChange, needs_original_node=True)
    def visit_TypeChange(self, orig, expr):
        # TypeChange conversions are supposed to keep the represented value intact, thus they
        # depend on the interpretation of the representation.
        if not isinstance(orig.type, MigenInteger) or not isinstance(orig.expr.type, MigenInteger):
            # so far, only conversions between MigenInteger is supported
            raise TypeError('Undefined (representation) conversion between types %s and %s' % (orig.expr.type, orig.type))
        conv = integer_repr_converters.get((orig.repr.ultimate_base, orig.expr.repr.ultimate_base))
        if conv is None:
            raise TypeError('Unknown conversion between integer representations %s and %s' % (orig.expr.repr, orig.repr))
        return conv(orig.repr, expr, orig.expr.repr)

    @visitor_for(TestIfNonzero, needs_original_node=True)
    def visit_TestIfNonzero(self, orig, expr):
        if not (
            (orig.expr.repr.ultimate_base == integer)
            and (orig.repr == bool)
        ):
            raise TypeError('VHDL TestIfNonzero works only on integer and returns boolean')
        return '('+expr + ' /= 0)'

    @visitor_for_wrapped(Signal,ClockSignal,ResetSignal)
    def visit_Signal(self, node):
        raise TypeError("Bare MyHDL signals should be replaced with VHDLSignals first: "+str(node.expr))

    def visit_Operator(self, node):
        migen_bits, migen_signed = value_bits_sign(node)
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
                type = (signed if migen_signed else unsigned)[migen_bits-1:0]
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




class ConverterOld:
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
        printer = VHDLExprPrinter(ns,_MapProxy(self.typeof), self._convert_expr_type)
        if type is not None and type is not True:
            return printer.visit_as_type(node,type)
        expr, etype = printer.visit(node)
        if type is True:
            return expr, etype
        return expr

    def _convert_expr_type(self, type, expr, orig_type):
        if orig_type.compatible_with(type):
            return expr
        if orig_type.castable_to(type) and type.ultimate_base.name is not None:
            return type.ultimate_base.name + '(' + expr + ')'
        converter = standard_type_conversions.get((type.ultimate_base,orig_type.ultimate_base),None)
        if converter is not None:
            return converter(type,expr,orig_type)
        if not isinstance(orig_type,VHDLArray) or not isinstance(type,VHDLArray):
            raise TypeError("Don't know how to convert type %s to %s"%(orig_type,type))
        if not orig_type.ultimate_base == type.ultimate_base:
            raise TypeError("Don't know how to convert type %s to %s"%(orig_type,type))
        # simply assume the presence of a resize function
        expr = 'resize('+expr+','+str(type.indextypes[0].length)+')'
        if not orig_type.valuetype.compatible_with(type.valuetype):
            assert orig_type.castable_to(type.valuetype)
            assert type.name is not None
            expr = type.name+'('+expr+')'
        return expr

    def _printnode(self, ns, level, node):
        if isinstance(node, _Assign):
            assignment = " <= "
            assert isinstance(node.l,(Signal,_Slice))
            left,leftt = self._printexpr(ns,node.l,type=True)
            return "\t" * level + left + assignment + self._printexpr(ns, node.r, type=leftt) + ";\n"
        elif isinstance(node, collections.Iterable):
            return "".join(self._printnode(ns, level, n) for n in node)
        elif isinstance(node, If):
            r = "\t" * level + "if " + self._printexpr(ns, node.cond, integer) + "/=0 then\n"
            r += self._printnode(ns, level + 1, node.t)
            if node.f:
                r += "\t" * level + "else\n"
                r += self._printnode(ns, level + 1, node.f)
            r += "\t" * level + "end if;\n"
            return r
        elif isinstance(node, Case):
            if node.cases:
                test,testt = self._printexpr(ns, node.test, type=True)

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
                condition = self._printexpr(ns, node.cond, integer) + '/=0'
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
use work.migen_helpers.all;

entity {name} is begin end {name};

architecture Migen of {name} is
component {dut}
\tport({dutport});
end component;
{signaldecl}
begin
dut: {dut} port map ({portmap});
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
            src += """clk_gen({name}_clk, {dt} ns, {advance} ns, {initial});\n""".format(
                name=k,
                dt=clk.half_period,
                advance=clk.half_period - clk.time_before_trans,
                initial="'1'" if clk.high else "'0'",
            )

        src += 'end;\n'

        return src

def convert(f, ios=None, name="top", special_overrides={}, create_clock_domains=True, asic_syntax=False):
    return Converter().convert(f,ios,name,special_overrides,create_clock_domains,asic_syntax)
