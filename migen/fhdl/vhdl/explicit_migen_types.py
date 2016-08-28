import abc

from ..structure import _Value

from ..visit_generic import (
    NodeTransformer as NodeTranformerGeneric,
    visitor_for, recursor_for, context_for, combiner_for,
)


# -------------------------------------------------------
# Type description classes
# -------------------------------------------------------

class AbstractExpressionType(abc.ABC):
    """ Abstract base class for the internal type hierarchy.

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


class MigenExpressionType(AbstractExpressionType):
    """ Abstract base class for Migen's front-end type API.

    Currently, Migen only supports integers: :py:class:`MigenInteger`
    """

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



# -----------------------------------------------------------
# AST expression node that contains explicit type information
# -----------------------------------------------------------

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

    Type information should be considered read-only:

    The annotated type is required to exactly match the type inferred from the
    semantics of the wrapped expression. If you want to change the semantics or
    restrict the type, inject a type-cast node.

    The same holds for the annotated representation: The representation field
    is set by the backend transformation to the repesentation the expession will
    actually have. If you require a different representation downstream,
    inject a type-cast.
    """
    def __init__(self,expr,type,repr=None):
        assert isinstance(expr,_Value) and not isinstance(expr,AbstractTypedExpression)
        AbstractTypedExpression.__init__(expr,type,repr)
        self.expr = expr

class TypeChange(AbstractTypedExpression):
    """ Replaces the representation and semantics of a type while preserving the
    represented value.

    Note that this might involve some logic in the underlying represenation or might
    simply be a type cast for the underlying representation.
    """

    def __init__(self, expr, *, type=None, repr=None):
        assert isinstance(expr, AbstractTypedExpression)
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


# -----------------------------------------------------------------------------
#  Extend NodeTransformer to handle explicit typing patch nodes MigenExpression
# -----------------------------------------------------------------------------

def recursor_for_wrapped(*wrapped_node_types):

    """ Decorator to register a method to handle nodes wrapped within :py:class:`MigenExpression` nodes.

    Any :py:class:`MigenExpression` node whose `expr` attribute matches any of the node classes given
    as argument to this decorator will be handled by the decorated function.
    """
    return NodeTranformerGeneric.registering_decorator(
        '_wrapped_node_recursors', wrapped_node_types,
    )

def combiner_for_wrapped(*wrapped_node_types):

    """ Decorator to register a method to handle nodes wrapped within :py:class:`MigenExpression` nodes.

    Any :py:class:`MigenExpression` node whose `expr` attribute matches any of the node classes given
    as argument to this decorator will be handled by the decorated function.
    """
    return NodeTranformerGeneric.registering_decorator(
        '_wrapped_node_combiners', wrapped_node_types,
    )

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

class NodeTransformer(NodeTranformerGeneric):
    """ Extends the NodeTransformer to allow forward explicit type information to the handlers for
    the implicitly typed Migen nodes.
    """

    @recursor_for(MigenExpression)
    def recurse_Annotated(self,node):
        handler = type(self).find_handler(
            '_wrapped_node_recursors',
            node.expr,
            type(self).recurse_unknown_wrapped_node
        )
        handler(self, node)

    def recurse_unknown_wrapped_node(self,node):
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
