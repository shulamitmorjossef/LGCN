from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from main import optimize_route, remove_driver_stop

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RouteRequest(BaseModel):
    driver_id: Optional[str] = None
    current_location: List[float]
    nodes: List[List[float]]
    end_point: Optional[List[float]] = None
    traffic_delays: Optional[List[List[float]]] = None
    traffic_matrix: Optional[List[List[float]]] = None  # alias sent by Flutter client
    priorities: Optional[List[float]] = None


@app.post("/compute-routes")
def compute_routes(request: RouteRequest):
    driver_id = request.driver_id or "default_driver"
    traffic = request.traffic_delays or request.traffic_matrix
    route_node_ids = optimize_route(
        driver_id=driver_id,
        current_location=request.current_location,
        stops=request.nodes,
        end_point=request.end_point,
        traffic_delays=traffic,
        priorities=request.priorities,
    )
    return {"status": "success", "routes": {"0": route_node_ids}}


@app.delete("/driver/{driver_id}")
def delete_driver(driver_id: str):
    from main import reset_driver
    reset_driver(driver_id)
    return {"status": "success", "driver_id": driver_id}


@app.delete("/driver/{driver_id}/stop")
def delete_driver_stop(driver_id: str, x: float, y: float):
    found = remove_driver_stop(driver_id, x, y)
    if not found:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Stop not found in driver cache")
    return {"status": "success", "driver_id": driver_id, "removed": {"x": x, "y": y}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
