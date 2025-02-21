from fastapi import HTTPException

class BadInputParsed(HTTPException):
    """Input is valid as long as it receives alphabetic strings"""
    
    def __init__(self, detail: str = "Invalid input. Only alphabetic strings are allowed"):
        super().__init__(status_code=400, detail=detail)

class MissingInput(HTTPException):
    """Input has not been provided"""
    
    def __init__(self, detail: str = "Invalid input. Please enter any sense string"):
        super().__init__(status_code=405, detail=detail)

class NoResultsException(HTTPException):
    """There were no results for searched criteria"""
    
    def __init__(self, detail: str = "No results. Your searching criteria doesn't exist on DBpedia!"):
        super().__init__(status_code=400, detail=detail)

# class TimeOutException(HTTPException):
#     """Time out exception occured. Site couldn't fetch the results in set time range"""
    
#     def __init__(self, detail: str = "No results. Time limit exceeded"):
#         super().__init__(status_code=413, detail=detail)      
