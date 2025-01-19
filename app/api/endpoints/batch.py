from fastapi import APIRouter, HTTPException, Request, Depends
from ...services.batch_service import BatchProcessingService
from ..dependencies import get_batch_processor

router = APIRouter()

@router.post("/start")
async def start_batch_processing(
    processor: BatchProcessingService = Depends(get_batch_processor)
):
    """Start the batch processing service"""
    try:
        await processor.start()
        return {"message": "Batch processor started successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_batch_processing(
    processor: BatchProcessingService = Depends(get_batch_processor)
):
    """Stop the batch processing service"""
    try:
        processor.stop()
        return {"message": "Batch processor stopped successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-once")
async def process_single_batch(
    processor: BatchProcessingService = Depends(get_batch_processor)
):
    """Process a single batch manually"""
    try:
        processed, errors = await processor.process_batch()
        return {
            "message": f"Processed {processed} listings with {errors} errors",
            "processed": processed,
            "errors": errors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_batch_processor(request: Request):
    """Get batch processor from app state"""
    return request.app.state.batch_service 