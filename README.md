# LGCN - Lightweight Graph Capsule Networks

## Overview

**LGCN** is a Python backend for single-driver route optimization. It solves the **VRPTW** using Capsule Networks and exposes results via a REST API consumed by a Flutter mobile client.



## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| Deep learning | PyTorch |
| API framework | FastAPI |
| ASGI server | Uvicorn |
| Data validation | Pydantic |
| Numerical computing | NumPy |
| State persistence | JSON files (per-driver cache) |

---

## Project Structure

```
LGCN/
│
├── server.py                          # FastAPI REST server (entry point for production)
├── main.py                            # Core routing logic, model init, driver state management
│
├── lgcn/
│   ├── lgcnModel.py                   # LGCN neural network architecture (capsule layers)
│   ├── capsule.py                     # Squash activation function
│   └── routing.py                     # Dynamic routing algorithm ("routing by agreement")
│
├── graph/
│   ├── node.py                        # Stop data structure (id, coordinates, time windows, wait)
│   ├── edge.py                        # Edge between stops (distance, traffic, urgency)
│   └── dynamicGraph.py                # Complete graph container — nodes and edges
│
├── baseLine/
│   └── greedy.py                      # Nearest-neighbor greedy baseline for comparison
│
├── driver_states/                     # Runtime-generated per-driver JSON state cache
└── lgcn_weights.pt                    # PyTorch model weights checkpoint
```

---

## API Reference

### `POST /compute-routes`

Compute the optimal route for a driver.

**Request body:**

```json
{
  "driver_id": "driver_42",
  "current_location": [32.0840, 34.7810, 0, 24, 0],
  "nodes": [
    [32.0853, 34.7818, 8.0, 12.0, 10],
    [32.0901, 34.7755, 9.0, 14.0, 5]
  ],
  "end_point": [32.0950, 34.7650, 0, 24, 0],
  "traffic_delays": [[1.0, 1.2], [1.1, 1.0]],
  "priorities": [1.0, 2.5]
}
```

Each node is a 5-element array: `[x, y, tw_start, tw_end, wait_minutes]`.

**Response:**

```json
{
  "status": "success",
  "routes": {
    "0": [-1, 2, 0, 1, -2]
  }
}
```

Node IDs: `-1` = current location, `-2` = end point, `≥ 0` = stop index in cache.

---

### `DELETE /driver/{driver_id}`

Clear the cached state for a driver (e.g., at the end of a shift).

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/shulamitmorjossef/LGCN.git
cd LGCN

# 2. Install dependencies
pip install torch numpy fastapi uvicorn pydantic
```

### Running the Server

```bash
python server.py
# OR
uvicorn server:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.


