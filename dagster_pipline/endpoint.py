from pydantic import BaseModel
from Dagster_pipline import generateuploader_form
from blacksheep import Application
import json
app = Application()
post = app.router.post



class order_details(BaseModel):
    order_number: str 

@post("/start/")
async def run_pipline(data: order_details):   # Note: Using the Pydantic model directly
    result = generateuploader_form.execute_in_process(run_config={"resources": {"order_data": {"config": data.dict()}}})
    
    validation_result = result.output_for_node("run_uploader_form")["validation_results"]
    breed_result = result.output_for_node("run_uploader_form")["breed_results"]

    # Modify the below return statement as per your requirements
    return {
        "validation": validation_result,
        "breed": breed_result
    }
