"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import dynamic from "next/dynamic";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
  ssr: false,
});

// ============================================================================
// Types
// ============================================================================

interface Paper {
  arxiv_id: string;
  title: string;
  abstract: string;
  authors: string[];
  published: string;
  pdf_url: string;
  categories: string[];
  cluster: number | null;
  cluster_name: string | null;
  x: number | null;
  y: number | null;
  neighbors: string[] | null; // arxiv_ids of nearest neighbors by embedding similarity
}

interface Category {
  id: number;
  name: string;
  description: string;
  color: string;
  count: number;
}

interface GraphNode {
  id: string;
  title: string;
  abstract: string;
  authors: string[];
  published: string;
  cluster: number;
  cluster_name: string;
  color: string;
  x: number;
  y: number;
  fx: number; // fixed x
  fy: number; // fixed y
}

interface GraphLink {
  source: string;
  target: string;
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

// ============================================================================
// API
// ============================================================================

function getApiUrl(): string {
  if (typeof window === "undefined") return "";
  // Dev mode: if not on port 8000, use backend at 8000
  const port = window.location.port;
  if (port !== "8000") return "http://localhost:8000";
  return "";
}

async function searchPapers(
  query: string,
  maxResults: number
): Promise<{ papers: Paper[]; categories: Category[] }> {
  const url = `${getApiUrl()}/api/search`;
  console.log("Fetching:", url, { query, max_results: maxResults });

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, max_results: maxResults }),
  });

  if (!res.ok) {
    const text = await res.text();
    console.error("API error:", res.status, text);
    throw new Error(`Search failed: ${res.status} ${text}`);
  }
  return res.json();
}

// ============================================================================
// Graph Data
// ============================================================================

function buildGraphData(papers: Paper[], categories: Category[]): GraphData {
  const colorMap = new Map(categories.map((c) => [c.id, c.color]));
  const nameMap = new Map(categories.map((c) => [c.id, c.name]));

  // Scale factor for positioning
  const scale = 300;

  const nodes: GraphNode[] = papers
    .filter((p) => p.x !== null && p.y !== null)
    .map((p) => ({
      id: p.arxiv_id,
      title: p.title,
      abstract: p.abstract,
      authors: p.authors,
      published: p.published,
      cluster: p.cluster!,
      cluster_name: p.cluster_name || nameMap.get(p.cluster!) || "Unknown",
      color: colorMap.get(p.cluster!) || "#666",
      x: p.x! * scale,
      y: p.y! * scale,
      fx: p.x! * scale, // Fix position
      fy: p.y! * scale,
    }));

  // Links based on nearest neighbors (embedding similarity)
  const links: GraphLink[] = [];
  const nodeIds = new Set(nodes.map((n) => n.id));

  papers.forEach((paper) => {
    if (!paper.neighbors) return;
    paper.neighbors.forEach((neighborId) => {
      // Only add link if neighbor exists in our nodes
      if (!nodeIds.has(neighborId)) return;
      // Avoid duplicate links (check both directions)
      if (!links.find((l) =>
        (l.source === paper.arxiv_id && l.target === neighborId) ||
        (l.source === neighborId && l.target === paper.arxiv_id)
      )) {
        links.push({ source: paper.arxiv_id, target: neighborId });
      }
    });
  });

  return { nodes, links };
}

// ============================================================================
// Paper Count Options
// ============================================================================

const PAPER_COUNTS = [50, 100, 200, 500, 1000];

// ============================================================================
// Main Component
// ============================================================================

export default function Home() {
  const [query, setQuery] = useState("");
  const [paperCount, setPaperCount] = useState(200);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState("");
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [categories, setCategories] = useState<Category[]>([]);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<any>(null);

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        });
      }
    };
    updateDimensions();
    window.addEventListener("resize", updateDimensions);
    return () => window.removeEventListener("resize", updateDimensions);
  }, []);

  const handleSearch = async () => {
    if (!query.trim() || isLoading) return;

    setIsLoading(true);
    setGraphData(null);
    setCategories([]);

    try {
      setLoadingStatus("Fetching papers...");
      const { papers, categories } = await searchPapers(query, paperCount);

      if (papers.length === 0) {
        setLoadingStatus("No papers found");
        setIsLoading(false);
        return;
      }

      setCategories(categories);
      const data = buildGraphData(papers, categories);
      setGraphData(data);
      setLoadingStatus("");

      // Center the graph
      setTimeout(() => {
        graphRef.current?.zoomToFit(400, 50);
      }, 100);
    } catch (error) {
      console.error("Search failed:", error);
      setLoadingStatus("Search failed");
    } finally {
      setIsLoading(false);
    }
  };

  const handleNodeHover = useCallback((node: GraphNode | null) => {
    setHoveredNode(node);
  }, []);

  // Seeded random for deterministic dendrites
  const seededRandom = (seed: number) => {
    const x = Math.sin(seed) * 10000;
    return x - Math.floor(x);
  };

  const nodeCanvasObject = useCallback(
    (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const isHovered = hoveredNode?.id === node.id;
      const baseSize = isHovered ? 6 : 4;

      // Draw glow on hover
      if (isHovered) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, baseSize * 2.5, 0, 2 * Math.PI);
        ctx.fillStyle = node.color + "22";
        ctx.fill();
      }

      // Draw node
      ctx.beginPath();
      ctx.arc(node.x, node.y, baseSize, 0, 2 * Math.PI);
      ctx.fillStyle = node.color;
      ctx.fill();

      // Inner highlight for 3D effect
      ctx.beginPath();
      ctx.arc(node.x - baseSize * 0.25, node.y - baseSize * 0.25, baseSize * 0.35, 0, 2 * Math.PI);
      ctx.fillStyle = "rgba(255,255,255,0.25)";
      ctx.fill();
    },
    [hoveredNode]
  );

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      {/* Left Sidebar */}
      <aside
        style={{
          width: 220,
          background: "var(--bg-secondary)",
          borderRight: "1px solid var(--border-subtle)",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
      >
        {/* Logo */}
        <div style={{ padding: "16px 20px", borderBottom: "1px solid var(--border-subtle)" }}>
          <h1 style={{ fontSize: "1.1rem", fontWeight: 600, color: "var(--text-primary)" }}>
            PaperMap
          </h1>
        </div>

        {/* Search */}
        <div style={{ padding: "16px" }}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            placeholder="Search arxiv..."
            disabled={isLoading}
            style={{
              width: "100%",
              padding: "10px 12px",
              background: "var(--bg-tertiary)",
              border: "1px solid var(--border-subtle)",
              borderRadius: "4px",
              color: "var(--text-primary)",
              fontSize: "0.9rem",
              outline: "none",
              marginBottom: "8px",
            }}
          />
          <div style={{ display: "flex", gap: "8px" }}>
            <select
              value={paperCount}
              onChange={(e) => setPaperCount(Number(e.target.value))}
              disabled={isLoading}
              style={{
                flex: 1,
                padding: "8px",
                background: "var(--bg-tertiary)",
                border: "1px solid var(--border-subtle)",
                borderRadius: "4px",
                color: "var(--text-primary)",
                fontSize: "0.8rem",
                cursor: "pointer",
              }}
            >
              {PAPER_COUNTS.map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
            <button
              onClick={handleSearch}
              disabled={isLoading || !query.trim()}
              style={{
                padding: "8px 14px",
                background: isLoading ? "var(--bg-tertiary)" : "var(--accent)",
                border: "none",
                borderRadius: "4px",
                color: "#fff",
                fontSize: "0.85rem",
                fontWeight: 500,
                cursor: isLoading ? "not-allowed" : "pointer",
                opacity: isLoading || !query.trim() ? 0.5 : 1,
              }}
            >
              {isLoading ? "..." : "Go"}
            </button>
          </div>
        </div>

        {/* Categories Legend */}
        <div style={{ flex: 1, overflow: "auto", padding: "0 16px 16px" }}>
          {categories.length > 0 && (
            <>
              <div
                style={{
                  fontSize: "0.7rem",
                  color: "var(--text-tertiary)",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  marginBottom: "12px",
                  fontWeight: 500,
                }}
              >
                Categories
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                {categories.map((cat) => (
                  <div
                    key={cat.id}
                    style={{
                      display: "flex",
                      alignItems: "flex-start",
                      gap: "10px",
                      padding: "8px 10px",
                      background: "var(--bg-tertiary)",
                      borderRadius: "4px",
                      cursor: "default",
                    }}
                    title={cat.description}
                  >
                    <span
                      style={{
                        width: 10,
                        height: 10,
                        borderRadius: "50%",
                        background: cat.color,
                        marginTop: 3,
                        flexShrink: 0,
                      }}
                    />
                    <div>
                      <div style={{ fontSize: "0.8rem", color: "var(--text-primary)", fontWeight: 500 }}>
                        {cat.name}
                      </div>
                      <div style={{ fontSize: "0.7rem", color: "var(--text-tertiary)", marginTop: 2 }}>
                        {cat.count} papers
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        {/* Graph */}
        <div ref={containerRef} style={{ flex: 1, position: "relative", overflow: "hidden" }}>
        {isLoading && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              gap: "12px",
              zIndex: 10,
            }}
          >
            <div
              className="loading-spin"
              style={{
                width: 28,
                height: 28,
                border: "3px solid var(--bg-tertiary)",
                borderTopColor: "var(--accent)",
                borderRadius: "50%",
              }}
            />
            <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>
              {loadingStatus}
            </p>
          </div>
        )}

        {!isLoading && !graphData && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              gap: "8px",
            }}
          >
            <p style={{ color: "var(--text-tertiary)", fontSize: "0.95rem" }}>
              Search to explore the paper landscape
            </p>
            <p style={{ color: "var(--text-tertiary)", fontSize: "0.8rem", fontFamily: "var(--font-mono)" }}>
              try: logit lens, sparse autoencoders, attention mechanism
            </p>
          </div>
        )}

        {graphData && !isLoading && (
          <ForceGraph2D
            ref={graphRef}
            graphData={graphData}
            width={dimensions.width}
            height={dimensions.height}
            nodeCanvasObject={nodeCanvasObject}
            nodePointerAreaPaint={(node: any, color, ctx) => {
              ctx.beginPath();
              ctx.arc(node.x, node.y, 12, 0, 2 * Math.PI);
              ctx.fillStyle = color;
              ctx.fill();
            }}
            linkCanvasObject={(link: any, ctx: CanvasRenderingContext2D) => {
              const source = link.source;
              const target = link.target;
              if (!source.x || !target.x) return;

              // Check if this link connects to hovered node
              const isHighlighted = hoveredNode &&
                (source.id === hoveredNode.id || target.id === hoveredNode.id);

              // Draw curved connection
              const midX = (source.x + target.x) / 2;
              const midY = (source.y + target.y) / 2;
              const dx = target.x - source.x;
              const dy = target.y - source.y;
              const dist = Math.sqrt(dx * dx + dy * dy + 0.01);
              const offset = dist * 0.1;
              const ctrlX = midX + (-dy / dist) * offset;
              const ctrlY = midY + (dx / dist) * offset;

              ctx.beginPath();
              ctx.moveTo(source.x, source.y);
              ctx.quadraticCurveTo(ctrlX, ctrlY, target.x, target.y);

              if (isHighlighted) {
                ctx.strokeStyle = hoveredNode!.color;
                ctx.lineWidth = 2;
                ctx.globalAlpha = 0.8;
              } else {
                ctx.strokeStyle = source.color + "15";
                ctx.lineWidth = 1;
                ctx.globalAlpha = 1;
              }
              ctx.stroke();
              ctx.globalAlpha = 1;
            }}
            onNodeHover={handleNodeHover}
            onNodeClick={(node: any) => {
              window.open(`https://arxiv.org/abs/${node.id}`, "_blank");
            }}
            cooldownTicks={0}
            enableNodeDrag={false}
            backgroundColor="transparent"
          />
        )}

        {/* Tooltip */}
        {hoveredNode && (
          <div
            style={{
              position: "absolute",
              bottom: 20,
              left: 20,
              right: 20,
              maxWidth: 550,
              padding: "14px 18px",
              background: "var(--bg-secondary)",
              border: "1px solid var(--border-medium)",
              borderRadius: "6px",
              boxShadow: "0 4px 20px rgba(0,0,0,0.4)",
            }}
          >
            <div style={{ display: "flex", gap: "10px" }}>
              <span
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: "50%",
                  background: hoveredNode.color,
                  marginTop: 5,
                  flexShrink: 0,
                }}
              />
              <div style={{ flex: 1, minWidth: 0 }}>
                <h3 style={{ fontSize: "0.9rem", fontWeight: 600, lineHeight: 1.4, marginBottom: 4 }}>
                  {hoveredNode.title}
                </h3>
                <p
                  style={{
                    fontSize: "0.75rem",
                    color: "var(--text-secondary)",
                    marginBottom: 6,
                    fontFamily: "var(--font-mono)",
                  }}
                >
                  {hoveredNode.authors.slice(0, 3).join(", ")}
                  {hoveredNode.authors.length > 3 && ` +${hoveredNode.authors.length - 3}`}
                  {" · "}
                  {new Date(hoveredNode.published).toLocaleDateString()}
                  {" · "}
                  {hoveredNode.cluster_name}
                </p>
                <p
                  style={{
                    fontSize: "0.8rem",
                    color: "var(--text-secondary)",
                    lineHeight: 1.5,
                    display: "-webkit-box",
                    WebkitLineClamp: 3,
                    WebkitBoxOrient: "vertical",
                    overflow: "hidden",
                  }}
                >
                  {hoveredNode.abstract}
                </p>
              </div>
            </div>
          </div>
        )}
        </div>
      </main>
    </div>
  );
}
