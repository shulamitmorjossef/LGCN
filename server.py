from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from main import optimize_route

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RouteRequest(BaseModel):
    driver_id: Optional[str] = None
    nodes: List[List[float]]
    traffic_delays: Optional[List[List[float]]] = None
    traffic_matrix: Optional[List[List[float]]] = None  # alias sent by Flutter client
    priorities: Optional[List[float]] = None


@app.post("/compute-routes")
def compute_routes(request: RouteRequest):
    driver_id = request.driver_id or "default_driver"
    traffic = request.traffic_delays or request.traffic_matrix
    route_node_ids = optimize_route(
        driver_id=driver_id,
        new_points=request.nodes,
        traffic_delays=traffic,
        priorities=request.priorities,
    )
    return {"status": "success", "routes": {"0": route_node_ids}}


@app.delete("/driver/{driver_id}")
def delete_driver(driver_id: str):
    from main import reset_driver
    reset_driver(driver_id)
    return {"status": "success", "driver_id": driver_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
