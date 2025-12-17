"use client";

import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import dynamic from "next/dynamic";
import { forceCollide } from "d3-force";

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
  neighbors: string[] | null;
  citation_count: number;
  references: string[] | null;
}

interface Category {
  id: number;
  name: string;
  description: string;
  color: string;
  count: number;
}

interface CitationLink {
  source: string;
  target: string;
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
  fx?: number;
  fy?: number;
  citation_count: number;
  pulse_intensity: number; // 0-1 normalized
  vx?: number;
  vy?: number;
}

interface GraphLink {
  source: string | GraphNode;
  target: string | GraphNode;
  isCitation?: boolean;
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
  const port = window.location.port;
  if (port !== "8000") return "http://localhost:8000";
  return "";
}

async function searchPapers(
  query: string,
  maxResults: number
): Promise<{
  papers: Paper[];
  categories: Category[];
  citation_links: CitationLink[];
  expanded_queries: string[];
  max_citations: number;
}> {
  const url = `${getApiUrl()}/api/search`;

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, max_results: maxResults }),
  });

  if (!res.ok) {
    throw new Error(`Search failed: ${res.status}`);
  }
  return res.json();
}

// ============================================================================
// Graph Data Builder
// ============================================================================

function buildGraphData(
  papers: Paper[],
  categories: Category[],
  citationLinks: CitationLink[],
  maxCitations: number
): GraphData {
  const colorMap = new Map(categories.map((c) => [c.id, c.color]));
  const nameMap = new Map(categories.map((c) => [c.id, c.name]));
  // Much larger scale for better spacing
  const scale = 600;

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
      // Don't fix positions - let force simulation add repulsion
      citation_count: p.citation_count,
      pulse_intensity: maxCitations > 0 ? Math.min(1, p.citation_count / maxCitations) : 0,
    }));

  const nodeIds = new Set(nodes.map((n) => n.id));
  const links: GraphLink[] = [];

  // Add similarity links
  papers.forEach((paper) => {
    if (!paper.neighbors) return;
    paper.neighbors.forEach((neighborId) => {
      if (!nodeIds.has(neighborId)) return;
      if (!links.find((l) => {
        const srcId = typeof l.source === 'string' ? l.source : l.source.id;
        const tgtId = typeof l.target === 'string' ? l.target : l.target.id;
        return (srcId === paper.arxiv_id && tgtId === neighborId) ||
               (srcId === neighborId && tgtId === paper.arxiv_id);
      })) {
        links.push({ source: paper.arxiv_id, target: neighborId, isCitation: false });
      }
    });
  });

  // Add citation links
  citationLinks.forEach((cl) => {
    if (nodeIds.has(cl.source) && nodeIds.has(cl.target)) {
      links.push({ source: cl.source, target: cl.target, isCitation: true });
    }
  });

  return { nodes, links };
}

// ============================================================================
// Constants
// ============================================================================

const PAPER_COUNTS = [50, 100, 200, 500];

// ============================================================================
// Component
// ============================================================================

export default function Home() {
  const [query, setQuery] = useState("");
  const [paperCount, setPaperCount] = useState(200);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState("");
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [categories, setCategories] = useState<Category[]>([]);
  const [expandedQueries, setExpandedQueries] = useState<string[]>([]);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [maxCitations, setMaxCitations] = useState(0);
  const [animationFrame, setAnimationFrame] = useState(0);

  // Filters
  const [dateFilter, setDateFilter] = useState<string>("all");
  const [visibleClusters, setVisibleClusters] = useState<Set<number>>(new Set());

  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<any>(null);

  // High-frequency animation loop for smooth plasma flow
  useEffect(() => {
    if (!graphData) return;
    let animId: number;
    let lastTime = 0;
    const animate = (time: number) => {
      if (time - lastTime > 16) { // ~60fps
        setAnimationFrame((f) => (f + 1) % 360);
        lastTime = time;
      }
      animId = requestAnimationFrame(animate);
    };
    animId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animId);
  }, [graphData]);

  // Date filter cutoffs
  const dateFilterCutoff = useMemo(() => {
    const now = new Date();
    switch (dateFilter) {
      case "1y": return new Date(now.getFullYear() - 1, now.getMonth(), now.getDate());
      case "2y": return new Date(now.getFullYear() - 2, now.getMonth(), now.getDate());
      case "5y": return new Date(now.getFullYear() - 5, now.getMonth(), now.getDate());
      default: return null;
    }
  }, [dateFilter]);

  // Filtered graph data
  const filteredGraphData = useMemo(() => {
    if (!graphData) return null;

    const filteredNodes = graphData.nodes.filter((node) => {
      if (!visibleClusters.has(node.cluster)) return false;
      if (dateFilterCutoff) {
        const paper = papers.find((p) => p.arxiv_id === node.id);
        if (paper) {
          const paperDate = new Date(paper.published);
          if (paperDate < dateFilterCutoff) return false;
        }
      }
      return true;
    });

    const nodeIds = new Set(filteredNodes.map((n) => n.id));
    const filteredLinks = graphData.links.filter((link) => {
      const sourceId = typeof link.source === "string" ? link.source : (link.source as any).id;
      const targetId = typeof link.target === "string" ? link.target : (link.target as any).id;
      return nodeIds.has(sourceId) && nodeIds.has(targetId);
    });

    return { nodes: filteredNodes, links: filteredLinks };
  }, [graphData, visibleClusters, dateFilterCutoff, papers]);

  const visibleCount = filteredGraphData?.nodes.length ?? 0;
  const totalCount = graphData?.nodes.length ?? 0;

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
    setPapers([]);
    setCategories([]);
    setExpandedQueries([]);
    setDateFilter("all");

    try {
      setLoadingStatus("Searching the web...");
      await new Promise((r) => setTimeout(r, 300));
      setLoadingStatus("Expanding search queries...");

      const result = await searchPapers(query, paperCount);

      if (result.papers.length === 0) {
        setLoadingStatus("No papers found");
        setIsLoading(false);
        return;
      }

      setExpandedQueries(result.expanded_queries);
      setPapers(result.papers);
      setCategories(result.categories);
      setMaxCitations(result.max_citations);
      setVisibleClusters(new Set(result.categories.map((c) => c.id)));

      setLoadingStatus("Building neural map...");
      const data = buildGraphData(
        result.papers,
        result.categories,
        result.citation_links,
        result.max_citations
      );
      setGraphData(data);
      setLoadingStatus("");

      // Configure forces and zoom after graph mounts
      setTimeout(() => {
        if (graphRef.current) {
          // Configure d3 forces for node spacing
          graphRef.current.d3Force("charge")?.strength(-400);
          graphRef.current.d3Force("link")?.distance(100).strength(0.3);

          // Add collision force to prevent overlap
          graphRef.current.d3Force(
            "collision",
            forceCollide().radius(25).strength(1)
          );

          // Reheat simulation
          graphRef.current.d3ReheatSimulation();
        }
      }, 100);

      setTimeout(() => {
        graphRef.current?.zoomToFit(400, 80);
      }, 500);
    } catch (error) {
      console.error("Search failed:", error);
      setLoadingStatus("Search failed");
    } finally {
      setIsLoading(false);
    }
  };

  const handleNodeHover = useCallback((node: any) => {
    setHoveredNode(node as GraphNode | null);
  }, []);

  // Bioluminescent node rendering - living neurons
  const nodeCanvasObject = useCallback(
    (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const isHovered = hoveredNode?.id === node.id;
      const time = animationFrame / 60;

      // Unique phase per node for organic feel
      const nodePhase = node.id.split('').reduce((a: number, c: string) => a + c.charCodeAt(0), 0) * 0.1;
      const breathe = Math.sin(time * Math.PI + nodePhase) * 0.5 + 0.5;

      // Dynamic sizing: base + citation boost + breathing
      const baseSize = 4;
      const citationBoost = node.pulse_intensity * 6;
      const hoverBoost = isHovered ? 4 : 0;
      const breatheBoost = breathe * (1 + node.pulse_intensity * 2);
      const size = baseSize + citationBoost + hoverBoost + breatheBoost;

      // === OUTER PLASMA FIELD (for high-citation nodes) ===
      if (node.pulse_intensity > 0.2 || isHovered) {
        const plasmaRadius = size * 3;
        const plasmaIntensity = isHovered ? 0.4 : 0.15 + node.pulse_intensity * 0.2;

        // Pulsing plasma ring
        const ringPhase = (time * 2 + nodePhase) % 1;
        const ringRadius = size * (1.5 + ringPhase * 1.5);
        const ringOpacity = (1 - ringPhase) * plasmaIntensity;

        ctx.beginPath();
        ctx.arc(node.x, node.y, ringRadius, 0, Math.PI * 2);
        ctx.strokeStyle = node.color;
        ctx.lineWidth = 2 * (1 - ringPhase);
        ctx.globalAlpha = ringOpacity;
        ctx.stroke();
        ctx.globalAlpha = 1;

        // Diffuse glow
        const glowGrad = ctx.createRadialGradient(node.x, node.y, size * 0.5, node.x, node.y, plasmaRadius);
        glowGrad.addColorStop(0, node.color + "60");
        glowGrad.addColorStop(0.4, node.color + "20");
        glowGrad.addColorStop(1, "transparent");
        ctx.beginPath();
        ctx.arc(node.x, node.y, plasmaRadius, 0, Math.PI * 2);
        ctx.fillStyle = glowGrad;
        ctx.fill();
      }

      // === CORONA EFFECT (hover or high citation) ===
      if (isHovered) {
        const coronaSize = size * 2.5;
        const coronaGrad = ctx.createRadialGradient(node.x, node.y, size, node.x, node.y, coronaSize);
        coronaGrad.addColorStop(0, "#00f5d4" + "80");
        coronaGrad.addColorStop(0.5, node.color + "40");
        coronaGrad.addColorStop(1, "transparent");
        ctx.beginPath();
        ctx.arc(node.x, node.y, coronaSize, 0, Math.PI * 2);
        ctx.fillStyle = coronaGrad;
        ctx.fill();
      }

      // === CORE NEURON ===
      // Outer edge glow
      const edgeGrad = ctx.createRadialGradient(node.x, node.y, size * 0.7, node.x, node.y, size * 1.2);
      edgeGrad.addColorStop(0, node.color);
      edgeGrad.addColorStop(0.8, node.color + "80");
      edgeGrad.addColorStop(1, "transparent");
      ctx.beginPath();
      ctx.arc(node.x, node.y, size * 1.2, 0, Math.PI * 2);
      ctx.fillStyle = edgeGrad;
      ctx.fill();

      // Solid core
      const coreGrad = ctx.createRadialGradient(
        node.x - size * 0.2, node.y - size * 0.2, 0,
        node.x, node.y, size
      );
      coreGrad.addColorStop(0, "#ffffff");
      coreGrad.addColorStop(0.2, node.color);
      coreGrad.addColorStop(0.8, node.color);
      coreGrad.addColorStop(1, node.color + "80");

      ctx.beginPath();
      ctx.arc(node.x, node.y, size, 0, Math.PI * 2);
      ctx.fillStyle = coreGrad;
      ctx.fill();

      // Specular highlight
      ctx.beginPath();
      ctx.arc(node.x - size * 0.3, node.y - size * 0.3, size * 0.35, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,255,255,0.6)";
      ctx.fill();
    },
    [hoveredNode, animationFrame]
  );

  // Helper: get point on quadratic bezier curve
  const getQuadraticPoint = (t: number, p0: {x: number, y: number}, p1: {x: number, y: number}, p2: {x: number, y: number}) => {
    const x = (1 - t) * (1 - t) * p0.x + 2 * (1 - t) * t * p1.x + t * t * p2.x;
    const y = (1 - t) * (1 - t) * p0.y + 2 * (1 - t) * t * p1.y + t * t * p2.y;
    return { x, y };
  };

  // Plasma synaptic link rendering with dramatic energy pulses
  const linkCanvasObject = useCallback(
    (link: any, ctx: CanvasRenderingContext2D) => {
      const source = link.source;
      const target = link.target;
      if (!source.x || !target.x) return;

      const isHighlighted = hoveredNode && (source.id === hoveredNode.id || target.id === hoveredNode.id);
      const isCitation = link.isCitation;
      const time = animationFrame / 60;

      // Curve geometry
      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const dist = Math.sqrt(dx * dx + dy * dy + 0.01);

      // Curved path with dynamic curvature
      const curvature = 0.15 + (isCitation ? 0.05 : 0);
      const midX = (source.x + target.x) / 2;
      const midY = (source.y + target.y) / 2;
      const ctrlX = midX + (-dy / dist) * dist * curvature;
      const ctrlY = midY + (dx / dist) * dist * curvature;

      const p0 = { x: source.x, y: source.y };
      const p1 = { x: ctrlX, y: ctrlY };
      const p2 = { x: target.x, y: target.y };

      // Unique seed for this link
      const linkSeed = (source.id + target.id).split("").reduce((a: number, c: string) => a + c.charCodeAt(0), 0);
      const linkPhase = (linkSeed % 100) / 100;

      // === BASE SYNAPSE LINE ===
      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.quadraticCurveTo(ctrlX, ctrlY, target.x, target.y);

      if (isHighlighted) {
        // Glowing highlighted line
        ctx.strokeStyle = isCitation ? "#00f5d4" : source.color;
        ctx.lineWidth = isCitation ? 3 : 2.5;
        ctx.globalAlpha = 1;
        ctx.shadowColor = isCitation ? "#00f5d4" : source.color;
        ctx.shadowBlur = 10;
      } else if (isCitation) {
        ctx.strokeStyle = "#00f5d4";
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.5;
      } else {
        ctx.strokeStyle = source.color;
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.15;
      }
      ctx.stroke();
      ctx.shadowBlur = 0;

      // === ENERGY PULSES ===
      // Multiple pulses traveling along the synapse
      const numPulses = isCitation ? 4 : 2;
      const pulseColor = isCitation ? "#00f5d4" : source.color;

      for (let i = 0; i < numPulses; i++) {
        // Each pulse travels at slightly different speed
        const baseSpeed = isCitation ? 0.8 : 0.4;
        const speedVar = 0.1 * (i % 2 === 0 ? 1 : 0.7);
        const speed = baseSpeed + speedVar;

        // Phase offset for staggered pulses
        const phase = ((time * speed) + (i / numPulses) + linkPhase) % 1;

        // Get position along curve
        const point = getQuadraticPoint(phase, p0, p1, p2);

        // Pulse size varies with position (bigger in middle)
        const positionFactor = Math.sin(phase * Math.PI);
        const basePulseSize = isCitation ? 5 : 3;
        const pulseSize = basePulseSize * (0.6 + positionFactor * 0.6);

        // Draw comet tail (trail behind the pulse)
        const tailLength = 8;
        ctx.beginPath();
        for (let t = 0; t < tailLength; t++) {
          const tailPhase = Math.max(0, phase - (t * 0.015));
          const tailPoint = getQuadraticPoint(tailPhase, p0, p1, p2);
          const tailOpacity = (1 - t / tailLength) * (isHighlighted ? 0.6 : (isCitation ? 0.4 : 0.2));

          ctx.globalAlpha = tailOpacity;
          ctx.beginPath();
          const tailSize = pulseSize * (1 - t / tailLength * 0.7);
          ctx.arc(tailPoint.x, tailPoint.y, tailSize, 0, Math.PI * 2);
          ctx.fillStyle = pulseColor;
          ctx.fill();
        }

        // === MAIN PULSE with intense glow ===
        // Outer halo
        const haloSize = pulseSize * 4;
        const haloGrad = ctx.createRadialGradient(point.x, point.y, 0, point.x, point.y, haloSize);
        haloGrad.addColorStop(0, pulseColor + "80");
        haloGrad.addColorStop(0.3, pulseColor + "30");
        haloGrad.addColorStop(1, "transparent");
        ctx.beginPath();
        ctx.arc(point.x, point.y, haloSize, 0, Math.PI * 2);
        ctx.fillStyle = haloGrad;
        ctx.globalAlpha = isHighlighted ? 0.9 : (isCitation ? 0.6 : 0.3);
        ctx.fill();

        // Core glow
        const coreGrad = ctx.createRadialGradient(point.x, point.y, 0, point.x, point.y, pulseSize * 1.5);
        coreGrad.addColorStop(0, "#ffffff");
        coreGrad.addColorStop(0.3, pulseColor);
        coreGrad.addColorStop(1, pulseColor + "00");
        ctx.beginPath();
        ctx.arc(point.x, point.y, pulseSize * 1.5, 0, Math.PI * 2);
        ctx.fillStyle = coreGrad;
        ctx.globalAlpha = isHighlighted ? 1 : (isCitation ? 0.8 : 0.5);
        ctx.fill();

        // Hot white center
        ctx.beginPath();
        ctx.arc(point.x, point.y, pulseSize * 0.4, 0, Math.PI * 2);
        ctx.fillStyle = "#ffffff";
        ctx.globalAlpha = isHighlighted ? 1 : (isCitation ? 0.9 : 0.6);
        ctx.fill();
      }

      ctx.globalAlpha = 1;
    },
    [hoveredNode, animationFrame]
  );

  // Styles
  const glassPanel: React.CSSProperties = {
    background: "rgba(15, 15, 25, 0.75)",
    backdropFilter: "blur(20px)",
    WebkitBackdropFilter: "blur(20px)",
    border: "1px solid rgba(255, 255, 255, 0.06)",
    boxShadow: "inset 0 1px 0 0 rgba(255,255,255,0.03), 0 25px 50px -12px rgba(0, 0, 0, 0.5)",
  };

  const inputStyle: React.CSSProperties = {
    width: "100%",
    padding: "12px 14px",
    background: "rgba(22, 22, 34, 0.8)",
    border: "1px solid rgba(255, 255, 255, 0.06)",
    borderRadius: "8px",
    color: "var(--text-primary)",
    fontSize: "0.9rem",
    outline: "none",
    transition: "all 0.2s ease",
  };

  return (
    <div style={{ display: "flex", height: "100vh", position: "relative", zIndex: 1 }}>
      {/* Left Sidebar */}
      <aside
        style={{
          ...glassPanel,
          width: 260,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
          borderRadius: 0,
          borderLeft: "none",
          borderTop: "none",
          borderBottom: "none",
        }}
      >
        {/* Logo */}
        <div
          style={{
            padding: "20px 24px",
            borderBottom: "1px solid rgba(255, 255, 255, 0.04)",
          }}
        >
          <h1
            style={{
              fontSize: "1.3rem",
              fontWeight: 600,
              background: "linear-gradient(135deg, #00f5d4 0%, #9b5de5 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              letterSpacing: "-0.02em",
            }}
          >
            PaperMap
          </h1>
          <p style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: 4, fontFamily: "var(--font-mono)" }}>
            Neural Research Explorer
          </p>
        </div>

        {/* Search */}
        <div style={{ padding: "20px" }}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            placeholder="Explore research..."
            disabled={isLoading}
            style={inputStyle}
          />
          <div style={{ display: "flex", gap: "10px", marginTop: "12px" }}>
            <select
              value={paperCount}
              onChange={(e) => setPaperCount(Number(e.target.value))}
              disabled={isLoading}
              style={{
                ...inputStyle,
                flex: 1,
                padding: "10px 12px",
                fontSize: "0.8rem",
                cursor: "pointer",
                appearance: "none",
              }}
            >
              {PAPER_COUNTS.map((n) => (
                <option key={n} value={n}>{n} papers</option>
              ))}
            </select>
            <button
              onClick={handleSearch}
              disabled={isLoading || !query.trim()}
              style={{
                padding: "10px 20px",
                background: isLoading
                  ? "rgba(22, 22, 34, 0.8)"
                  : "linear-gradient(135deg, #00f5d4 0%, #00b4d8 100%)",
                border: "none",
                borderRadius: "8px",
                color: isLoading ? "var(--text-muted)" : "#000",
                fontSize: "0.85rem",
                fontWeight: 600,
                cursor: isLoading ? "not-allowed" : "pointer",
                opacity: isLoading || !query.trim() ? 0.5 : 1,
                transition: "all 0.2s ease",
              }}
            >
              {isLoading ? "..." : "Go"}
            </button>
          </div>
        </div>

        {/* Content */}
        <div style={{ flex: 1, overflow: "auto", padding: "0 20px 20px" }}>
          {/* Expanded Queries */}
          {expandedQueries.length > 0 && (
            <div style={{ marginBottom: "24px" }}>
              <div
                style={{
                  fontSize: "0.65rem",
                  color: "var(--text-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.1em",
                  marginBottom: "10px",
                  fontWeight: 500,
                }}
              >
                Search Expansion
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                {expandedQueries.map((q, i) => (
                  <span
                    key={i}
                    style={{
                      fontSize: "0.65rem",
                      padding: "4px 8px",
                      background: "rgba(0, 245, 212, 0.08)",
                      border: "1px solid rgba(0, 245, 212, 0.15)",
                      borderRadius: "4px",
                      color: "var(--text-secondary)",
                      fontFamily: "var(--font-mono)",
                    }}
                  >
                    {q}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Categories */}
          {categories.length > 0 && (
            <div>
              <div
                style={{
                  fontSize: "0.65rem",
                  color: "var(--text-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.1em",
                  marginBottom: "12px",
                  fontWeight: 500,
                }}
              >
                Clusters
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                {categories.map((cat) => (
                  <div
                    key={cat.id}
                    style={{
                      display: "flex",
                      alignItems: "flex-start",
                      gap: "12px",
                      padding: "12px",
                      background: "rgba(22, 22, 34, 0.5)",
                      borderRadius: "8px",
                      border: "1px solid rgba(255, 255, 255, 0.03)",
                      cursor: "default",
                      transition: "all 0.2s ease",
                    }}
                    title={cat.description}
                  >
                    <span
                      style={{
                        width: 10,
                        height: 10,
                        borderRadius: "50%",
                        background: cat.color,
                        marginTop: 4,
                        flexShrink: 0,
                        boxShadow: `0 0 10px ${cat.color}40`,
                      }}
                    />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{
                        fontSize: "0.8rem",
                        color: "var(--text-primary)",
                        fontWeight: 500,
                        lineHeight: 1.3,
                      }}>
                        {cat.name}
                      </div>
                      <div style={{
                        fontSize: "0.7rem",
                        color: "var(--text-muted)",
                        marginTop: 3,
                        fontFamily: "var(--font-mono)",
                      }}>
                        {cat.count} papers
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <div ref={containerRef} style={{ flex: 1, position: "relative", overflow: "hidden" }}>
          {/* Loading State */}
          {isLoading && (
            <div
              style={{
                position: "absolute",
                inset: 0,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                gap: "16px",
                zIndex: 10,
              }}
            >
              <div
                style={{
                  width: 40,
                  height: 40,
                  border: "2px solid rgba(0, 245, 212, 0.2)",
                  borderTopColor: "#00f5d4",
                  borderRadius: "50%",
                  animation: "spin 1s linear infinite",
                }}
              />
              <p style={{
                color: "var(--text-secondary)",
                fontSize: "0.9rem",
                fontFamily: "var(--font-mono)",
              }}>
                {loadingStatus}
              </p>
            </div>
          )}

          {/* Empty State */}
          {!isLoading && !graphData && (
            <div
              style={{
                position: "absolute",
                inset: 0,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                gap: "12px",
              }}
            >
              <div
                style={{
                  width: 80,
                  height: 80,
                  borderRadius: "50%",
                  background: "radial-gradient(circle, rgba(0, 245, 212, 0.1) 0%, transparent 70%)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  marginBottom: 8,
                }}
              >
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#00f5d4" strokeWidth="1.5">
                  <circle cx="12" cy="12" r="3" />
                  <path d="M12 2v4m0 12v4m10-10h-4M6 12H2m15.07-7.07l-2.83 2.83m-8.48 8.48l-2.83 2.83m14.14 0l-2.83-2.83M6.34 6.34L3.51 3.51" />
                </svg>
              </div>
              <p style={{ color: "var(--text-secondary)", fontSize: "1rem" }}>
                Explore the research landscape
              </p>
              <p style={{
                color: "var(--text-muted)",
                fontSize: "0.8rem",
                fontFamily: "var(--font-mono)",
              }}>
                try: flash attention, logit lens, sparse autoencoders
              </p>
            </div>
          )}

          {/* Graph */}
          {filteredGraphData && !isLoading && (
            <ForceGraph2D
              ref={graphRef}
              graphData={filteredGraphData}
              width={dimensions.width}
              height={dimensions.height}
              nodeCanvasObject={nodeCanvasObject}
              nodePointerAreaPaint={(node: any, color, ctx) => {
                // Larger hit area for easier hovering
                const hitRadius = 20 + (node.pulse_intensity || 0) * 10;
                ctx.beginPath();
                ctx.arc(node.x, node.y, hitRadius, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
              }}
              linkCanvasObject={linkCanvasObject}
              onNodeHover={handleNodeHover}
              onNodeClick={(node: any) => {
                window.open(`https://arxiv.org/abs/${node.id}`, "_blank");
              }}
              // Force simulation for organic node spacing
              d3AlphaDecay={0.015}
              d3VelocityDecay={0.25}
              cooldownTime={4000}
              warmupTicks={150}
              enableNodeDrag={true}
              backgroundColor="transparent"
            />
          )}

          {/* Tooltip */}
          {hoveredNode && (
            <div
              style={{
                ...glassPanel,
                position: "absolute",
                bottom: 24,
                left: 24,
                right: 24,
                maxWidth: 600,
                padding: "18px 22px",
                borderRadius: "12px",
                animation: "fade-in-up 0.2s ease-out",
              }}
            >
              <div style={{ display: "flex", gap: "14px" }}>
                <div
                  style={{
                    width: 12,
                    height: 12,
                    borderRadius: "50%",
                    background: hoveredNode.color,
                    marginTop: 5,
                    flexShrink: 0,
                    boxShadow: `0 0 12px ${hoveredNode.color}60`,
                  }}
                />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <h3 style={{
                    fontSize: "0.95rem",
                    fontWeight: 600,
                    lineHeight: 1.4,
                    marginBottom: 6,
                  }}>
                    {hoveredNode.title}
                  </h3>
                  <p style={{
                    fontSize: "0.75rem",
                    color: "var(--text-secondary)",
                    marginBottom: 8,
                    fontFamily: "var(--font-mono)",
                    display: "flex",
                    flexWrap: "wrap",
                    gap: "6px",
                    alignItems: "center",
                  }}>
                    <span>{hoveredNode.authors.slice(0, 3).join(", ")}{hoveredNode.authors.length > 3 && ` +${hoveredNode.authors.length - 3}`}</span>
                    <span style={{ color: "var(--text-muted)" }}>·</span>
                    <span>{new Date(hoveredNode.published).toLocaleDateString()}</span>
                    <span style={{ color: "var(--text-muted)" }}>·</span>
                    <span style={{ color: hoveredNode.color }}>{hoveredNode.cluster_name}</span>
                    {hoveredNode.citation_count > 0 && (
                      <>
                        <span style={{ color: "var(--text-muted)" }}>·</span>
                        <span style={{ color: "#00f5d4" }}>{hoveredNode.citation_count} citations</span>
                      </>
                    )}
                  </p>
                  <p style={{
                    fontSize: "0.8rem",
                    color: "var(--text-secondary)",
                    lineHeight: 1.6,
                    display: "-webkit-box",
                    WebkitLineClamp: 3,
                    WebkitBoxOrient: "vertical",
                    overflow: "hidden",
                  }}>
                    {hoveredNode.abstract}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Right Sidebar - Filters */}
      {graphData && (
        <aside
          style={{
            ...glassPanel,
            width: 220,
            padding: "20px",
            display: "flex",
            flexDirection: "column",
            gap: "24px",
            overflow: "auto",
            borderRadius: 0,
            borderRight: "none",
            borderTop: "none",
            borderBottom: "none",
          }}
        >
          {/* Stats */}
          <div>
            <div style={{
              fontSize: "2rem",
              fontWeight: 700,
              color: "#00f5d4",
              lineHeight: 1,
            }}>
              {visibleCount}
            </div>
            <div style={{
              fontSize: "0.7rem",
              color: "var(--text-muted)",
              marginTop: 4,
              fontFamily: "var(--font-mono)",
            }}>
              of {totalCount} papers visible
            </div>
          </div>

          {/* Date Filter */}
          <div>
            <div style={{
              fontSize: "0.65rem",
              color: "var(--text-muted)",
              textTransform: "uppercase",
              letterSpacing: "0.1em",
              marginBottom: "12px",
              fontWeight: 500,
            }}>
              Time Range
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              {[
                { value: "all", label: "All time" },
                { value: "5y", label: "Last 5 years" },
                { value: "2y", label: "Last 2 years" },
                { value: "1y", label: "Last year" },
              ].map((opt) => (
                <label
                  key={opt.value}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "10px",
                    padding: "8px 10px",
                    borderRadius: "6px",
                    fontSize: "0.8rem",
                    color: dateFilter === opt.value ? "var(--text-primary)" : "var(--text-secondary)",
                    background: dateFilter === opt.value ? "rgba(0, 245, 212, 0.1)" : "transparent",
                    cursor: "pointer",
                    transition: "all 0.15s ease",
                  }}
                >
                  <input
                    type="radio"
                    name="dateFilter"
                    value={opt.value}
                    checked={dateFilter === opt.value}
                    onChange={(e) => setDateFilter(e.target.value)}
                    style={{ display: "none" }}
                  />
                  <div
                    style={{
                      width: 14,
                      height: 14,
                      borderRadius: "50%",
                      border: `2px solid ${dateFilter === opt.value ? "#00f5d4" : "var(--text-muted)"}`,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    {dateFilter === opt.value && (
                      <div style={{
                        width: 6,
                        height: 6,
                        borderRadius: "50%",
                        background: "#00f5d4",
                      }} />
                    )}
                  </div>
                  {opt.label}
                </label>
              ))}
            </div>
          </div>

          {/* Cluster Toggle */}
          <div>
            <div style={{
              fontSize: "0.65rem",
              color: "var(--text-muted)",
              textTransform: "uppercase",
              letterSpacing: "0.1em",
              marginBottom: "12px",
              fontWeight: 500,
            }}>
              Clusters
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              {categories.map((cat) => (
                <label
                  key={cat.id}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "10px",
                    padding: "8px 10px",
                    borderRadius: "6px",
                    fontSize: "0.75rem",
                    color: visibleClusters.has(cat.id) ? "var(--text-primary)" : "var(--text-muted)",
                    background: visibleClusters.has(cat.id) ? "rgba(255,255,255,0.03)" : "transparent",
                    cursor: "pointer",
                    transition: "all 0.15s ease",
                  }}
                >
                  <div
                    onClick={(e) => {
                      e.preventDefault();
                      const newSet = new Set(visibleClusters);
                      if (newSet.has(cat.id)) {
                        newSet.delete(cat.id);
                      } else {
                        newSet.add(cat.id);
                      }
                      setVisibleClusters(newSet);
                    }}
                    style={{
                      width: 16,
                      height: 16,
                      borderRadius: "4px",
                      border: `2px solid ${visibleClusters.has(cat.id) ? cat.color : "var(--text-muted)"}`,
                      background: visibleClusters.has(cat.id) ? cat.color : "transparent",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      transition: "all 0.15s ease",
                    }}
                  >
                    {visibleClusters.has(cat.id) && (
                      <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                        <path d="M2 5l2 2 4-4" stroke="#000" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    )}
                  </div>
                  <span style={{
                    flex: 1,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}>
                    {cat.name}
                  </span>
                </label>
              ))}
            </div>
          </div>

          {/* Legend */}
          <div style={{ marginTop: "auto", paddingTop: 16, borderTop: "1px solid rgba(255,255,255,0.04)" }}>
            <div style={{
              fontSize: "0.65rem",
              color: "var(--text-muted)",
              textTransform: "uppercase",
              letterSpacing: "0.1em",
              marginBottom: "10px",
              fontWeight: 500,
            }}>
              Visual Key
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "0.7rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{
                  width: 12,
                  height: 12,
                  borderRadius: "50%",
                  background: "#00f5d4",
                  boxShadow: "0 0 8px #00f5d440",
                }} />
                <span style={{ color: "var(--text-secondary)" }}>High citations</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{
                  width: 20,
                  height: 2,
                  background: "#00f5d466",
                  borderRadius: 1,
                }} />
                <span style={{ color: "var(--text-secondary)" }}>Citation link</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{
                  width: 20,
                  height: 1,
                  background: "rgba(255,255,255,0.15)",
                  borderRadius: 1,
                }} />
                <span style={{ color: "var(--text-secondary)" }}>Similarity link</span>
              </div>
            </div>
          </div>
        </aside>
      )}
    </div>
  );
}
