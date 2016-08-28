""" fhdl.vhdl -- Convert/generate and manipulate VHDL using Migen.

VHDL is a strongly typed language, and it's type system is very strict, with almost no implicit type casts.
Due to this, a major part of this sub-package deals with handling explicit types.

In Migen, we therefore add two levels of explicit typing:
1. Semantic types: Explicitely describe how values behave in Migen.
2. Representation: How these values are represented in a back-end, in particular VHDL.

- :py:modl:`.explicit_migen_types` provides the tools to explicitely represent the first layer (semantic types)
 in Migen, including a subclass to :py:class:`..visit_generic.NodeTransformer` that handles the additional
 AST nodes.
- :py:modl:`.ast` then adds additional AST nodes that explicitely deal with the second layer (representation),
 allowing to explicitely encode representation changes in the AST. Again, a subclass to
 :py:class:`.explicit_migen_types.NodeTransformer` is provided, which handles the new nodes.
- :py:modl:`.types` contains tools to represent and manipulate VHDL types (i.e. VHDL specific representations of Migen values).
- :py:modl:`.type_annotator` supplies :py:class:`.type_annotator.ExplicitTyper`, a node transfomrmer that adds
 explicit semantic types to a Migen AST without explicit such information.
- :py:modl:`.lowerer` contains a node transformer that transforms a Migen AST into an AST closely matching
 a VHDL ast, including annotation of actual representations.
- :py:modl:`.writer` generates VHDL code from suitable ASTs.
- :py:modl:`.converter` wraps everything into a single easy to use interface

TODO:
 - Refactor integer ranges for consistency (appears in Migen slices, semantic migen types and VHDL types)
"""


from .converter import Converter
