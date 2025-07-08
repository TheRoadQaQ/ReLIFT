# Generated from D:/Code/GitProjects/uniform-eval/src/uniform_eval/bundled/latex2sympy2/assets\PS.g4 by ANTLR 4.11.1
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .PSParser import PSParser
else:
    from PSParser import PSParser

# This class defines a complete listener for a parse tree produced by PSParser.
class PSListener(ParseTreeListener):

    # Enter a parse tree produced by PSParser#accent_symbol.
    def enterAccent_symbol(self, ctx:PSParser.Accent_symbolContext):
        pass

    # Exit a parse tree produced by PSParser#accent_symbol.
    def exitAccent_symbol(self, ctx:PSParser.Accent_symbolContext):
        pass


    # Enter a parse tree produced by PSParser#math.
    def enterMath(self, ctx:PSParser.MathContext):
        pass

    # Exit a parse tree produced by PSParser#math.
    def exitMath(self, ctx:PSParser.MathContext):
        pass


    # Enter a parse tree produced by PSParser#transpose.
    def enterTranspose(self, ctx:PSParser.TransposeContext):
        pass

    # Exit a parse tree produced by PSParser#transpose.
    def exitTranspose(self, ctx:PSParser.TransposeContext):
        pass


    # Enter a parse tree produced by PSParser#transform_atom.
    def enterTransform_atom(self, ctx:PSParser.Transform_atomContext):
        pass

    # Exit a parse tree produced by PSParser#transform_atom.
    def exitTransform_atom(self, ctx:PSParser.Transform_atomContext):
        pass


    # Enter a parse tree produced by PSParser#transform_scale.
    def enterTransform_scale(self, ctx:PSParser.Transform_scaleContext):
        pass

    # Exit a parse tree produced by PSParser#transform_scale.
    def exitTransform_scale(self, ctx:PSParser.Transform_scaleContext):
        pass


    # Enter a parse tree produced by PSParser#transform_swap.
    def enterTransform_swap(self, ctx:PSParser.Transform_swapContext):
        pass

    # Exit a parse tree produced by PSParser#transform_swap.
    def exitTransform_swap(self, ctx:PSParser.Transform_swapContext):
        pass


    # Enter a parse tree produced by PSParser#transform_assignment.
    def enterTransform_assignment(self, ctx:PSParser.Transform_assignmentContext):
        pass

    # Exit a parse tree produced by PSParser#transform_assignment.
    def exitTransform_assignment(self, ctx:PSParser.Transform_assignmentContext):
        pass


    # Enter a parse tree produced by PSParser#elementary_transform.
    def enterElementary_transform(self, ctx:PSParser.Elementary_transformContext):
        pass

    # Exit a parse tree produced by PSParser#elementary_transform.
    def exitElementary_transform(self, ctx:PSParser.Elementary_transformContext):
        pass


    # Enter a parse tree produced by PSParser#elementary_transforms.
    def enterElementary_transforms(self, ctx:PSParser.Elementary_transformsContext):
        pass

    # Exit a parse tree produced by PSParser#elementary_transforms.
    def exitElementary_transforms(self, ctx:PSParser.Elementary_transformsContext):
        pass


    # Enter a parse tree produced by PSParser#matrix.
    def enterMatrix(self, ctx:PSParser.MatrixContext):
        pass

    # Exit a parse tree produced by PSParser#matrix.
    def exitMatrix(self, ctx:PSParser.MatrixContext):
        pass


    # Enter a parse tree produced by PSParser#det.
    def enterDet(self, ctx:PSParser.DetContext):
        pass

    # Exit a parse tree produced by PSParser#det.
    def exitDet(self, ctx:PSParser.DetContext):
        pass


    # Enter a parse tree produced by PSParser#matrix_row.
    def enterMatrix_row(self, ctx:PSParser.Matrix_rowContext):
        pass

    # Exit a parse tree produced by PSParser#matrix_row.
    def exitMatrix_row(self, ctx:PSParser.Matrix_rowContext):
        pass


    # Enter a parse tree produced by PSParser#relation.
    def enterRelation(self, ctx:PSParser.RelationContext):
        pass

    # Exit a parse tree produced by PSParser#relation.
    def exitRelation(self, ctx:PSParser.RelationContext):
        pass


    # Enter a parse tree produced by PSParser#relation_list.
    def enterRelation_list(self, ctx:PSParser.Relation_listContext):
        pass

    # Exit a parse tree produced by PSParser#relation_list.
    def exitRelation_list(self, ctx:PSParser.Relation_listContext):
        pass


    # Enter a parse tree produced by PSParser#relation_list_content.
    def enterRelation_list_content(self, ctx:PSParser.Relation_list_contentContext):
        pass

    # Exit a parse tree produced by PSParser#relation_list_content.
    def exitRelation_list_content(self, ctx:PSParser.Relation_list_contentContext):
        pass


    # Enter a parse tree produced by PSParser#equality.
    def enterEquality(self, ctx:PSParser.EqualityContext):
        pass

    # Exit a parse tree produced by PSParser#equality.
    def exitEquality(self, ctx:PSParser.EqualityContext):
        pass


    # Enter a parse tree produced by PSParser#expr.
    def enterExpr(self, ctx:PSParser.ExprContext):
        pass

    # Exit a parse tree produced by PSParser#expr.
    def exitExpr(self, ctx:PSParser.ExprContext):
        pass


    # Enter a parse tree produced by PSParser#additive.
    def enterAdditive(self, ctx:PSParser.AdditiveContext):
        pass

    # Exit a parse tree produced by PSParser#additive.
    def exitAdditive(self, ctx:PSParser.AdditiveContext):
        pass


    # Enter a parse tree produced by PSParser#mp.
    def enterMp(self, ctx:PSParser.MpContext):
        pass

    # Exit a parse tree produced by PSParser#mp.
    def exitMp(self, ctx:PSParser.MpContext):
        pass


    # Enter a parse tree produced by PSParser#mp_nofunc.
    def enterMp_nofunc(self, ctx:PSParser.Mp_nofuncContext):
        pass

    # Exit a parse tree produced by PSParser#mp_nofunc.
    def exitMp_nofunc(self, ctx:PSParser.Mp_nofuncContext):
        pass


    # Enter a parse tree produced by PSParser#unary.
    def enterUnary(self, ctx:PSParser.UnaryContext):
        pass

    # Exit a parse tree produced by PSParser#unary.
    def exitUnary(self, ctx:PSParser.UnaryContext):
        pass


    # Enter a parse tree produced by PSParser#unary_nofunc.
    def enterUnary_nofunc(self, ctx:PSParser.Unary_nofuncContext):
        pass

    # Exit a parse tree produced by PSParser#unary_nofunc.
    def exitUnary_nofunc(self, ctx:PSParser.Unary_nofuncContext):
        pass


    # Enter a parse tree produced by PSParser#postfix.
    def enterPostfix(self, ctx:PSParser.PostfixContext):
        pass

    # Exit a parse tree produced by PSParser#postfix.
    def exitPostfix(self, ctx:PSParser.PostfixContext):
        pass


    # Enter a parse tree produced by PSParser#postfix_nofunc.
    def enterPostfix_nofunc(self, ctx:PSParser.Postfix_nofuncContext):
        pass

    # Exit a parse tree produced by PSParser#postfix_nofunc.
    def exitPostfix_nofunc(self, ctx:PSParser.Postfix_nofuncContext):
        pass


    # Enter a parse tree produced by PSParser#postfix_op.
    def enterPostfix_op(self, ctx:PSParser.Postfix_opContext):
        pass

    # Exit a parse tree produced by PSParser#postfix_op.
    def exitPostfix_op(self, ctx:PSParser.Postfix_opContext):
        pass


    # Enter a parse tree produced by PSParser#eval_at.
    def enterEval_at(self, ctx:PSParser.Eval_atContext):
        pass

    # Exit a parse tree produced by PSParser#eval_at.
    def exitEval_at(self, ctx:PSParser.Eval_atContext):
        pass


    # Enter a parse tree produced by PSParser#eval_at_sub.
    def enterEval_at_sub(self, ctx:PSParser.Eval_at_subContext):
        pass

    # Exit a parse tree produced by PSParser#eval_at_sub.
    def exitEval_at_sub(self, ctx:PSParser.Eval_at_subContext):
        pass


    # Enter a parse tree produced by PSParser#eval_at_sup.
    def enterEval_at_sup(self, ctx:PSParser.Eval_at_supContext):
        pass

    # Exit a parse tree produced by PSParser#eval_at_sup.
    def exitEval_at_sup(self, ctx:PSParser.Eval_at_supContext):
        pass


    # Enter a parse tree produced by PSParser#exp.
    def enterExp(self, ctx:PSParser.ExpContext):
        pass

    # Exit a parse tree produced by PSParser#exp.
    def exitExp(self, ctx:PSParser.ExpContext):
        pass


    # Enter a parse tree produced by PSParser#exp_nofunc.
    def enterExp_nofunc(self, ctx:PSParser.Exp_nofuncContext):
        pass

    # Exit a parse tree produced by PSParser#exp_nofunc.
    def exitExp_nofunc(self, ctx:PSParser.Exp_nofuncContext):
        pass


    # Enter a parse tree produced by PSParser#comp.
    def enterComp(self, ctx:PSParser.CompContext):
        pass

    # Exit a parse tree produced by PSParser#comp.
    def exitComp(self, ctx:PSParser.CompContext):
        pass


    # Enter a parse tree produced by PSParser#comp_nofunc.
    def enterComp_nofunc(self, ctx:PSParser.Comp_nofuncContext):
        pass

    # Exit a parse tree produced by PSParser#comp_nofunc.
    def exitComp_nofunc(self, ctx:PSParser.Comp_nofuncContext):
        pass


    # Enter a parse tree produced by PSParser#group.
    def enterGroup(self, ctx:PSParser.GroupContext):
        pass

    # Exit a parse tree produced by PSParser#group.
    def exitGroup(self, ctx:PSParser.GroupContext):
        pass


    # Enter a parse tree produced by PSParser#norm_group.
    def enterNorm_group(self, ctx:PSParser.Norm_groupContext):
        pass

    # Exit a parse tree produced by PSParser#norm_group.
    def exitNorm_group(self, ctx:PSParser.Norm_groupContext):
        pass


    # Enter a parse tree produced by PSParser#abs_group.
    def enterAbs_group(self, ctx:PSParser.Abs_groupContext):
        pass

    # Exit a parse tree produced by PSParser#abs_group.
    def exitAbs_group(self, ctx:PSParser.Abs_groupContext):
        pass


    # Enter a parse tree produced by PSParser#floor_group.
    def enterFloor_group(self, ctx:PSParser.Floor_groupContext):
        pass

    # Exit a parse tree produced by PSParser#floor_group.
    def exitFloor_group(self, ctx:PSParser.Floor_groupContext):
        pass


    # Enter a parse tree produced by PSParser#ceil_group.
    def enterCeil_group(self, ctx:PSParser.Ceil_groupContext):
        pass

    # Exit a parse tree produced by PSParser#ceil_group.
    def exitCeil_group(self, ctx:PSParser.Ceil_groupContext):
        pass


    # Enter a parse tree produced by PSParser#accent.
    def enterAccent(self, ctx:PSParser.AccentContext):
        pass

    # Exit a parse tree produced by PSParser#accent.
    def exitAccent(self, ctx:PSParser.AccentContext):
        pass


    # Enter a parse tree produced by PSParser#atom_expr_no_supexpr.
    def enterAtom_expr_no_supexpr(self, ctx:PSParser.Atom_expr_no_supexprContext):
        pass

    # Exit a parse tree produced by PSParser#atom_expr_no_supexpr.
    def exitAtom_expr_no_supexpr(self, ctx:PSParser.Atom_expr_no_supexprContext):
        pass


    # Enter a parse tree produced by PSParser#atom_expr.
    def enterAtom_expr(self, ctx:PSParser.Atom_exprContext):
        pass

    # Exit a parse tree produced by PSParser#atom_expr.
    def exitAtom_expr(self, ctx:PSParser.Atom_exprContext):
        pass


    # Enter a parse tree produced by PSParser#atom.
    def enterAtom(self, ctx:PSParser.AtomContext):
        pass

    # Exit a parse tree produced by PSParser#atom.
    def exitAtom(self, ctx:PSParser.AtomContext):
        pass


    # Enter a parse tree produced by PSParser#mathit.
    def enterMathit(self, ctx:PSParser.MathitContext):
        pass

    # Exit a parse tree produced by PSParser#mathit.
    def exitMathit(self, ctx:PSParser.MathitContext):
        pass


    # Enter a parse tree produced by PSParser#mathit_text.
    def enterMathit_text(self, ctx:PSParser.Mathit_textContext):
        pass

    # Exit a parse tree produced by PSParser#mathit_text.
    def exitMathit_text(self, ctx:PSParser.Mathit_textContext):
        pass


    # Enter a parse tree produced by PSParser#frac.
    def enterFrac(self, ctx:PSParser.FracContext):
        pass

    # Exit a parse tree produced by PSParser#frac.
    def exitFrac(self, ctx:PSParser.FracContext):
        pass


    # Enter a parse tree produced by PSParser#binom.
    def enterBinom(self, ctx:PSParser.BinomContext):
        pass

    # Exit a parse tree produced by PSParser#binom.
    def exitBinom(self, ctx:PSParser.BinomContext):
        pass


    # Enter a parse tree produced by PSParser#func_normal_functions_single_arg.
    def enterFunc_normal_functions_single_arg(self, ctx:PSParser.Func_normal_functions_single_argContext):
        pass

    # Exit a parse tree produced by PSParser#func_normal_functions_single_arg.
    def exitFunc_normal_functions_single_arg(self, ctx:PSParser.Func_normal_functions_single_argContext):
        pass


    # Enter a parse tree produced by PSParser#func_normal_functions_multi_arg.
    def enterFunc_normal_functions_multi_arg(self, ctx:PSParser.Func_normal_functions_multi_argContext):
        pass

    # Exit a parse tree produced by PSParser#func_normal_functions_multi_arg.
    def exitFunc_normal_functions_multi_arg(self, ctx:PSParser.Func_normal_functions_multi_argContext):
        pass


    # Enter a parse tree produced by PSParser#func_operator_names_single_arg.
    def enterFunc_operator_names_single_arg(self, ctx:PSParser.Func_operator_names_single_argContext):
        pass

    # Exit a parse tree produced by PSParser#func_operator_names_single_arg.
    def exitFunc_operator_names_single_arg(self, ctx:PSParser.Func_operator_names_single_argContext):
        pass


    # Enter a parse tree produced by PSParser#func_operator_names_multi_arg.
    def enterFunc_operator_names_multi_arg(self, ctx:PSParser.Func_operator_names_multi_argContext):
        pass

    # Exit a parse tree produced by PSParser#func_operator_names_multi_arg.
    def exitFunc_operator_names_multi_arg(self, ctx:PSParser.Func_operator_names_multi_argContext):
        pass


    # Enter a parse tree produced by PSParser#func_normal_single_arg.
    def enterFunc_normal_single_arg(self, ctx:PSParser.Func_normal_single_argContext):
        pass

    # Exit a parse tree produced by PSParser#func_normal_single_arg.
    def exitFunc_normal_single_arg(self, ctx:PSParser.Func_normal_single_argContext):
        pass


    # Enter a parse tree produced by PSParser#func_normal_multi_arg.
    def enterFunc_normal_multi_arg(self, ctx:PSParser.Func_normal_multi_argContext):
        pass

    # Exit a parse tree produced by PSParser#func_normal_multi_arg.
    def exitFunc_normal_multi_arg(self, ctx:PSParser.Func_normal_multi_argContext):
        pass


    # Enter a parse tree produced by PSParser#func.
    def enterFunc(self, ctx:PSParser.FuncContext):
        pass

    # Exit a parse tree produced by PSParser#func.
    def exitFunc(self, ctx:PSParser.FuncContext):
        pass


    # Enter a parse tree produced by PSParser#args.
    def enterArgs(self, ctx:PSParser.ArgsContext):
        pass

    # Exit a parse tree produced by PSParser#args.
    def exitArgs(self, ctx:PSParser.ArgsContext):
        pass


    # Enter a parse tree produced by PSParser#func_common_args.
    def enterFunc_common_args(self, ctx:PSParser.Func_common_argsContext):
        pass

    # Exit a parse tree produced by PSParser#func_common_args.
    def exitFunc_common_args(self, ctx:PSParser.Func_common_argsContext):
        pass


    # Enter a parse tree produced by PSParser#limit_sub.
    def enterLimit_sub(self, ctx:PSParser.Limit_subContext):
        pass

    # Exit a parse tree produced by PSParser#limit_sub.
    def exitLimit_sub(self, ctx:PSParser.Limit_subContext):
        pass


    # Enter a parse tree produced by PSParser#func_single_arg.
    def enterFunc_single_arg(self, ctx:PSParser.Func_single_argContext):
        pass

    # Exit a parse tree produced by PSParser#func_single_arg.
    def exitFunc_single_arg(self, ctx:PSParser.Func_single_argContext):
        pass


    # Enter a parse tree produced by PSParser#func_single_arg_noparens.
    def enterFunc_single_arg_noparens(self, ctx:PSParser.Func_single_arg_noparensContext):
        pass

    # Exit a parse tree produced by PSParser#func_single_arg_noparens.
    def exitFunc_single_arg_noparens(self, ctx:PSParser.Func_single_arg_noparensContext):
        pass


    # Enter a parse tree produced by PSParser#func_multi_arg.
    def enterFunc_multi_arg(self, ctx:PSParser.Func_multi_argContext):
        pass

    # Exit a parse tree produced by PSParser#func_multi_arg.
    def exitFunc_multi_arg(self, ctx:PSParser.Func_multi_argContext):
        pass


    # Enter a parse tree produced by PSParser#func_multi_arg_noparens.
    def enterFunc_multi_arg_noparens(self, ctx:PSParser.Func_multi_arg_noparensContext):
        pass

    # Exit a parse tree produced by PSParser#func_multi_arg_noparens.
    def exitFunc_multi_arg_noparens(self, ctx:PSParser.Func_multi_arg_noparensContext):
        pass


    # Enter a parse tree produced by PSParser#subexpr.
    def enterSubexpr(self, ctx:PSParser.SubexprContext):
        pass

    # Exit a parse tree produced by PSParser#subexpr.
    def exitSubexpr(self, ctx:PSParser.SubexprContext):
        pass


    # Enter a parse tree produced by PSParser#supexpr.
    def enterSupexpr(self, ctx:PSParser.SupexprContext):
        pass

    # Exit a parse tree produced by PSParser#supexpr.
    def exitSupexpr(self, ctx:PSParser.SupexprContext):
        pass


    # Enter a parse tree produced by PSParser#subeq.
    def enterSubeq(self, ctx:PSParser.SubeqContext):
        pass

    # Exit a parse tree produced by PSParser#subeq.
    def exitSubeq(self, ctx:PSParser.SubeqContext):
        pass


    # Enter a parse tree produced by PSParser#supeq.
    def enterSupeq(self, ctx:PSParser.SupeqContext):
        pass

    # Exit a parse tree produced by PSParser#supeq.
    def exitSupeq(self, ctx:PSParser.SupeqContext):
        pass



del PSParser