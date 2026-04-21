from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
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
    num_drivers: int = 1
    driver_starts: List[int] = [0]


@app.post("/compute-routes")
def compute_routes(request: RouteRequest):
    route_indices = optimize_route(request.nodes)
    return {"status": "success", "routes": {"0": route_indices}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
