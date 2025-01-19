from fastapi import Request

def get_batch_processor(request: Request):
    """Get batch processor from app state"""
    return request.app.state.batch_service 