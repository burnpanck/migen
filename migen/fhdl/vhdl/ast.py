""" vhdl.ast -- Extensions to Migen's AST to cater for the needs of VHDL

"""

import abc

from ..structure import _Statement, _Assign, _Value
from ..visit_generic import recursor_for, combiner_for, visitor_for
from .explicit_migen_types import AbstractExpressionType, AbstractTypedExpression, TypeChange
from .explicit_migen_types import NodeTransformer as ExplicitlyTypedNodeTransformer
from .syntax import reserved_keywords
# ----------------------------------------------------------------------------
# Additional semantic types not part of Migen's front-end, but needed for VHDL
# ----------------------------------------------------------------------------

class Boolean(AbstractExpressionType):
    """ Explicit boolean, required for arguments of conditional statements.

    Migen's conditionals implicitely test for non-zeroness of the condition.
    Since VHDL doesn't, when transforming the AST for VHDL, we want to make this
    test explicit. However, since the Migen semantics for comparison operations
    produce an integer (0 or 1), we need a way to distinguish the semantics
    of the result of a comparison operator in Migen and the result of a nonzero test
    in a conditional.

    TODO: instead we could keep that test for non-zeroness implicit and just
    generate that test at the last moment when writing the VHDL.
    That would however exclude the possibilty to implement optimisers at the AST
    level that e.g. remove redundant tests.
    """

    @property
    def nbits(self):
        return 1

    def compatible_with(self, other):
        if isinstance(other, Boolean):
            return True
        return False

    def contained_in(self, other):
        if isinstance(other, Boolean):
            return True
        return False

class TestIfNonzero(TypeChange):
    """ Explicit test for nonzero integer values, returning a boolean result.

    Note that migen's comparison operators return integers (either 0 or 1).
    If statements test for non-zero integers, while assigning to 1-bit registers
    truncates to the LSB.
    """

    def __init__(self, expr, *, repr=None):
        super().__init__(
            expr,
            type=Boolean(),
            repr=repr
        )


# ---------------------------------------------
# AST nodes to represent VHDL specific elements
# ---------------------------------------------


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
    k:ReservedKeyword for k in reserved_keywords
})

class Entity(Scope):
    def __init__(self,name,ports=[]):
        assert all(isinstance(p,Port) for p in ports)
        self.name = name
        self.ports = ports

class Component:
    # TODO: generics
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
        assert all(isinstance(v,(Signal,VHDLSignal)) for k,v in portmap.items())
        self.name = name
        self.component = component
        self.portmap = portmap

class EntityBody(Scope):
    def __init__(self,entity,*,statements=[],architecture='Migen',signals={},components={},namespace={}):
        """
        :param entity: The entity which is implemented in this body
        :param architecture: The architecture name
        :param signals: The local signals (ports are defined on `entity`)
        :param components: Declarations of components
        :param namespace: Mapping of all names to their meaning inside the body.
        :param statements: The (concurrent) statements (including processes and component instantiations) contained in the body.
        """
        assert isinstance(entity,Entity)
        assert all(isinstance(s,VHDLSignal) for s in signals.values())
        if not isinstance(components, dict):
            components = {c.name:c for c in components}
        assert all(isinstance(i,Component) for i in components.values())
        self.architecture = architecture
        self.entity = entity
        self.statements = statements
        self.signals = signals
        self.components = components
        self.namespace = namespace

    @property
    def parent_scope(self):
        return self.entity

    def _get_object(self,name):
        return self.signals.get(name,None) or self.components.get(name,None)



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


# ---------------------------------------------
# NodeTransform extension for VHDL nodes
# ---------------------------------------------

class NodeTransformer(ExplicitlyTypedNodeTransformer):
    @combiner_for(TestIfNonzero)
    def combine_TestIfNonzero(self, node, expr, *, type, repr):
        assert type == Boolean()
        return type(node)(expr, repr=repr)

    @recursor_for(ConditionalAssignment)
    def recurse_ConditionalAssignment(self, node):
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
    def recurse_SelectedAssignment(self, node):
        raise NotImplementedError('recurse_SelectedAssignment')