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
    nodes: List[List[float]]
    traffic_delays: Optional[List[List[float]]] = None
    priorities: Optional[List[float]] = None


@app.post("/compute-routes")
def compute_routes(request: RouteRequest):
    route_indices = optimize_route(
        request.nodes,
        traffic_delays=request.traffic_delays,
        priorities=request.priorities,
    )
    return {"status": "success", "routes": {"0": route_indices}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
