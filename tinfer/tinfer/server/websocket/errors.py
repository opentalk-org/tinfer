from dataclasses import dataclass

from aiohttp import web


@dataclass(frozen=True)
class ValidationIssue:
    loc: tuple[str | int, ...]
    msg: str
    type: str


class RequestValidationError(ValueError):
    def __init__(self, issue: ValidationIssue) -> None:
        super().__init__(issue.msg)
        self.issue = issue


def validation_response(error: RequestValidationError) -> web.Response:
    issue = error.issue
    return web.json_response(
        {
            "detail": [
                {"loc": list(issue.loc), "msg": issue.msg, "type": issue.type}
            ]
        },
        status=422,
    )
