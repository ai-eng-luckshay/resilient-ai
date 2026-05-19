"""Calculate tool — safe AST-based mathematical expression evaluator."""
import ast
import logging
import math
import operator

from langchain_core.tools import tool

from app.config.logging_config import get_trace_id
from app.config.metrics import get_metrics

logger = logging.getLogger(__name__)

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        return _SAFE_OPS[op_type](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval(node.operand)
    if isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        allowed_funcs = {"sqrt": math.sqrt, "abs": abs, "round": round}
        if func_name in allowed_funcs:
            args = [_safe_eval(a) for a in node.args]
            return allowed_funcs[func_name](*args)
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


@tool
async def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Supports +, -, *, /, **, %, sqrt(), abs(), round()."""
    trace = get_trace_id()
    logger.info("[%s] tool=calculate expr=%r", trace, expression)
    get_metrics().record_tool_call("calculate")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree.body)
        return f"{expression} = {result}"
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"
