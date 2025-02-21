from fastapi import FastAPI
from fastapi import Request, status
from router import anarchy_of_hierarchy, cohesion_among_categories
from exceptions import BadInputParsed, NoResultsException, MissingInput
from fastapi.responses import  JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Callable
from loguru import logger
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND, HTTP_405_METHOD_NOT_ALLOWED, HTTP_413_REQUEST_ENTITY_TOO_LARGE



app = FastAPI(title="Analiza hijerarhije i slicnosti koncepata iz DBpedia baze")
app.include_router(anarchy_of_hierarchy.router)
app.include_router(cohesion_among_categories.router)

@app.exception_handler(MissingInput)
async def missing_input_exception_handler(request: Request, exc: MissingInput):
    return JSONResponse(
        status_code=HTTP_405_METHOD_NOT_ALLOWED,
        content={"detail": exc.detail},
    )
# @app.exception_handler(TimeOutException)
# async def timeout_exception_handler(request: Request, exc: TimeOutException):
#     return JSONResponse(
#         status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE,
#         content={"detail": exc.detail},
#     )

@app.exception_handler(BadInputParsed)
async def bad_input_exception_handler(request: Request, exc: BadInputParsed):
    return JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content={"detail": exc.detail},
    )

@app.exception_handler(NoResultsException)
async def no_results_found_exception_handler(request: Request, exc: NoResultsException):
    return JSONResponse(
        status_code=HTTP_404_NOT_FOUND,
        content={"detail": exc.detail},
    )


# if __name__=='__main__':
