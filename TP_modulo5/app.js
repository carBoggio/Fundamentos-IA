// ─── Mazes ───────────────────────────────────────────────────────────────────

const MAZES = [
  {
    name: "Laberinto 1 (15×15)",
    start: [1, 0],
    goal:  [13, 14],
    grid: [
      [0,0,0,1,0,0,0,1,0,0,0,0,0,1,0],
      [0,1,0,1,0,1,0,1,0,1,1,1,0,1,0],
      [0,1,0,0,0,1,0,0,0,0,0,1,0,0,0],
      [0,1,1,1,1,1,1,1,1,1,0,1,1,1,0],
      [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
      [1,1,1,1,1,1,1,0,1,1,1,1,1,1,0],
      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
      [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
      [0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
      [0,1,0,1,1,1,1,1,1,1,1,1,0,1,0],
      [0,1,0,1,0,0,0,0,0,0,0,1,0,1,0],
      [0,1,0,1,0,1,1,1,1,1,0,1,0,1,0],
      [0,1,0,1,0,1,0,0,0,1,0,1,0,1,0],
      [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
      [1,1,0,1,1,1,1,1,1,1,1,1,1,1,0],
    ],
  },
  {
    name: "Laberinto 2 (15×15)",
    start: [0, 0],
    goal:  [14, 14],
    grid: [
      [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
      [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
      [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
      [1,1,1,1,1,1,0,1,1,1,1,1,1,1,0],
      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
      [0,1,1,1,0,1,1,1,0,1,1,1,0,1,0],
      [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
      [1,1,0,1,1,1,0,1,1,1,0,1,1,1,0],
      [0,0,0,0,0,1,0,0,0,1,0,0,0,0,0],
      [0,1,1,1,0,1,1,1,0,1,1,1,0,1,0],
      [0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
      [0,1,0,1,1,1,1,1,1,1,1,1,0,1,0],
      [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
      [1,1,0,1,0,1,1,1,1,1,0,1,0,1,1],
      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ],
  },
  {
    name: "Laberinto 3 (25×25)",
    start: [0, 0],
    goal:  [24, 24],
    grid: [
      [0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0],
      [0,1,0,1,0,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,0],
      [0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0],
      [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,0],
      [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0],
      [1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0],
      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
      [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
      [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
      [0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0],
      [0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0],
      [0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0],
      [0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0],
      [0,0,0,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,0,0,0],
      [1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0],
      [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
      [0,1,1,1,0,1,1,1,0,1,0,0,0,0,0,1,0,1,1,1,0,1,1,1,0],
      [0,0,0,1,0,0,0,1,0,1,0,1,1,1,0,1,0,0,0,1,0,0,0,1,0],
      [1,1,0,1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1,0],
      [0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0],
      [0,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,0],
      [0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
      [0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
      [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
      [1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0],
    ],
  },
];

// ─── Directions ──────────────────────────────────────────────────────────────

const DIRS = [[-1,0],[1,0],[0,-1],[0,1]];

function neighbors(grid, [r, c]) {
  const result = [];
  for (const [dr, dc] of DIRS) {
    const nr = r + dr, nc = c + dc;
    if (nr >= 0 && nr < grid.length && nc >= 0 && nc < grid[0].length && grid[nr][nc] === 0) {
      result.push([nr, nc]);
    }
  }
  return result;
}

function key([r, c]) { return `${r},${c}`; }

function reconstructPath(cameFrom, start, goal) {
  const path = [];
  let cur = goal;
  while (key(cur) !== key(start)) {
    path.unshift(cur);
    cur = cameFrom.get(key(cur));
    if (!cur) return [];
  }
  path.unshift(start);
  return path;
}

// ─── Heuristics & cost ───────────────────────────────────────────────────────

function heuristic([r1, c1], [r2, c2]) {
  return Math.abs(r1 - r2) + Math.abs(c1 - c2);
}

function getCost(cameFrom, node) {
  let cost = 0;
  let cur = node;
  while (cameFrom.has(key(cur))) {
    cost++;
    cur = cameFrom.get(key(cur));
  }
  return cost;
}

// ─── BFS ─────────────────────────────────────────────────────────────────────

function bfs(grid, start, goal) {
  const queue = [start];
  const visited = new Set([key(start)]);
  const cameFrom = new Map();
  const visitedOrder = [];

  while (queue.length > 0) {
    const current = queue.shift();
    visitedOrder.push(current);

    if (key(current) === key(goal)) {
      return { path: reconstructPath(cameFrom, start, goal), visitedOrder, nodesExpanded: visitedOrder.length };
    }

    for (const nb of neighbors(grid, current)) {
      if (!visited.has(key(nb))) {
        visited.add(key(nb));
        cameFrom.set(key(nb), current);
        queue.push(nb);
      }
    }
  }

  return { path: [], visitedOrder, nodesExpanded: visitedOrder.length };
}

// ─── DFS ─────────────────────────────────────────────────────────────────────

function dfs(grid, start, goal) {
  const stack = [start];
  const visited = new Set([key(start)]);
  const cameFrom = new Map();
  const visitedOrder = [];

  while (stack.length > 0) {
    const current = stack.pop();
    visitedOrder.push(current);

    if (key(current) === key(goal)) {
      return { path: reconstructPath(cameFrom, start, goal), visitedOrder, nodesExpanded: visitedOrder.length };
    }

    for (const nb of neighbors(grid, current)) {
      if (!visited.has(key(nb))) {
        visited.add(key(nb));
        cameFrom.set(key(nb), current);
        stack.push(nb);
      }
    }
  }

  return { path: [], visitedOrder, nodesExpanded: visitedOrder.length };
}

// ─── A* ──────────────────────────────────────────────────────────────────────

function aStar(grid, start, goal) {
  // Simple array-based priority queue (small grids — fine for this scale)
  const open = [{ node: start, f: heuristic(start, goal) }];
  const cameFrom = new Map();
  const gScore = new Map([[key(start), 0]]);
  const visitedOrder = [];
  const closed = new Set();

  while (open.length > 0) {
    open.sort((a, b) => a.f - b.f);
    const { node: current } = open.shift();
    const ck = key(current);

    if (closed.has(ck)) continue;
    closed.add(ck);
    visitedOrder.push(current);

    if (ck === key(goal)) {
      return { path: reconstructPath(cameFrom, start, goal), visitedOrder, nodesExpanded: visitedOrder.length };
    }

    for (const nb of neighbors(grid, current)) {
      const nk = key(nb);
      if (closed.has(nk)) continue;

      const tentativeG = (gScore.get(ck) ?? Infinity) + 1;
      if (tentativeG < (gScore.get(nk) ?? Infinity)) {
        gScore.set(nk, tentativeG);
        cameFrom.set(nk, current);
        open.push({ node: nb, f: tentativeG + heuristic(nb, goal) });
      }
    }
  }

  return { path: [], visitedOrder, nodesExpanded: visitedOrder.length };
}
