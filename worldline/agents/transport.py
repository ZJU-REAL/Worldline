"""Model-specific error classification over the common HTTP runtime."""

from worldline.errors import HTTPRequestError, ModelAPIError
from worldline.runtime.http import http_json


def http_completion(request, deadline, max_bytes, cancel_check=None):
    try:
        return http_json(request, deadline, max_bytes, cancel_check=cancel_check)
    except HTTPRequestError as error:
        raise ModelAPIError(str(error), status=error.status, retry_after=error.retry_after) from error
