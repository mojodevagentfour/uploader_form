from pydantic import BaseModel
from Dagster_pipline import generatekohya_lora
from blacksheep import Application
import json
app = Application()
post = app.router.post



class order_details(BaseModel):
    order_number: str 
    animal_type : str
    breed: str

@post("/start/")
async def run_pipline(data: order_details):   # Note: Using the Pydantic model directly
    result = generatekohya_lora.execute_in_process(run_config={"resources": {"order_data": {"config": data.dict()}}})
    # image_validation_result = result.output_for_node("image_validation")
    # if image_validation_result:
    #     return {"result": image_validation_result["validation_result"]}
    
    return True

