"use client";

import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import dynamic from "next/dynamic";
import { forceCollide } from "d3-force";
import Markdown from "react-markdown";

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
  // Dev: frontend on 3000, backend on 8000
  if (port === "3000") return "http://localhost:8000";
  // Production: same origin (frontend served by backend)
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
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [maxCitations, setMaxCitations] = useState(0);
  const [animationFrame, setAnimationFrame] = useState(0);

  // Filters
  const [dateFilter, setDateFilter] = useState<string>("all");
  const [visibleClusters, setVisibleClusters] = useState<Set<number>>(new Set());

  // UI State
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<Array<{ role: 'user' | 'assistant'; content: string }>>([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [webSearchEnabled, setWebSearchEnabled] = useState(true);

  // Synaptic Context Dock state
  const [contextPapers, setContextPapers] = useState<GraphNode[]>([]);
  const [draggingNode, setDraggingNode] = useState<GraphNode | null>(null);
  const [dragScreenPos, setDragScreenPos] = useState<{ x: number; y: number } | null>(null);
  const [isNearDock, setIsNearDock] = useState(false);
  const [parsedPapers, setParsedPapers] = useState<Set<string>>(new Set()); // Track which papers have been parsed
  const [parsingPapers, setParsingPapers] = useState<Set<string>>(new Set()); // Track papers currently being parsed
  const dragStartPos = useRef<{ x: number; y: number } | null>(null);
  const draggingNodeRef = useRef<GraphNode | null>(null);
  const isNearDockRef = useRef(false);
  const prevContextPapersRef = useRef<string[]>([]); // Track previous papers for change detection

  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<any>(null);
  const dockZoneHeight = 100; // Drop zone height in pixels

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

  // Track context paper changes - announce and pre-parse
  useEffect(() => {
    const currentIds = contextPapers.map(p => p.id);
    const prevIds = prevContextPapersRef.current;

    // Find added and removed papers
    const added = contextPapers.filter(p => !prevIds.includes(p.id));
    const removed = prevIds.filter(id => !currentIds.includes(id));

    // Announce changes in chat (only if chat has started)
    if (chatMessages.length > 0 || chatOpen) {
      added.forEach(paper => {
        setChatMessages(prev => [...prev, {
          role: 'assistant' as const,
          content: `📄 **Paper added to context:** ${paper.title}`
        }]);
      });

      removed.forEach(id => {
        const paper = papers.find(p => p.arxiv_id === id);
        if (paper) {
          setChatMessages(prev => [...prev, {
            role: 'assistant' as const,
            content: `📄 **Paper removed from context:** ${paper.title}`
          }]);
        }
      });
    }

    // Pre-parse newly added papers with polling
    added.forEach(async (paper) => {
      if (parsedPapers.has(paper.id) || parsingPapers.has(paper.id)) return;

      // Mark as parsing
      setParsingPapers(prev => new Set([...prev, paper.id]));

      try {
        const fullPaper = papers.find(p => p.arxiv_id === paper.id);
        const pdfUrl = fullPaper?.pdf_url || `https://arxiv.org/pdf/${paper.id}.pdf`;

        // Start parse job
        const response = await fetch(`${getApiUrl()}/api/parse-paper`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            arxiv_id: paper.id,
            pdf_url: pdfUrl,
          }),
        });

        if (!response.ok) throw new Error('Failed to start parse job');

        const data = await response.json();

        // If cached, mark as parsed immediately
        if (data.cached || data.status === 'completed') {
          setParsedPapers(prev => new Set([...prev, paper.id]));
          setParsingPapers(prev => {
            const next = new Set(prev);
            next.delete(paper.id);
            return next;
          });
          return;
        }

        // Poll for status
        const jobId = data.job_id;
        const pollInterval = 2000; // 2 seconds
        const maxPolls = 300; // 10 minutes max

        for (let i = 0; i < maxPolls; i++) {
          await new Promise(resolve => setTimeout(resolve, pollInterval));

          const statusRes = await fetch(`${getApiUrl()}/api/parse-status/${jobId}`);
          if (!statusRes.ok) throw new Error('Failed to get parse status');

          const status = await statusRes.json();

          if (status.status === 'completed') {
            setParsedPapers(prev => new Set([...prev, paper.id]));
            break;
          } else if (status.status === 'failed') {
            console.error(`Parse failed for ${paper.id}: ${status.error}`);
            break;
          }
          // Still processing, continue polling
        }
      } catch (error) {
        console.error(`Failed to pre-parse paper ${paper.id}:`, error);
      } finally {
        setParsingPapers(prev => {
          const next = new Set(prev);
          next.delete(paper.id);
          return next;
        });
      }
    });

    // Update ref
    prevContextPapersRef.current = currentIds;
  }, [contextPapers, papers, chatMessages.length, chatOpen, parsedPapers, parsingPapers]);

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
    if (node && graphRef.current) {
      const coords = graphRef.current.graph2ScreenCoords(node.x, node.y);
      if (coords) {
        setTooltipPos({ x: coords.x, y: coords.y });
      }
    } else {
      setTooltipPos(null);
    }
  }, []);

  // Handle drag end via mouseup listener
  useEffect(() => {
    const handleMouseUp = () => {
      const node = draggingNodeRef.current;
      if (!node) return;

      // Check if dropped in dock zone
      if (isNearDockRef.current) {
        // Add to context if not already there
        setContextPapers(prev => {
          if (prev.find(p => p.id === node.id)) return prev;
          return [...prev, node];
        });
      }

      // Reset drag state
      setDraggingNode(null);
      setDragScreenPos(null);
      setIsNearDock(false);
      draggingNodeRef.current = null;
      isNearDockRef.current = false;
      dragStartPos.current = null;
    };

    window.addEventListener("mouseup", handleMouseUp);
    return () => window.removeEventListener("mouseup", handleMouseUp);
  }, []);

  const removeFromContext = useCallback((paperId: string) => {
    setContextPapers(prev => prev.filter(p => p.id !== paperId));
  }, []);

  // Send chat message to API
  const sendChatMessage = useCallback(async (message: string) => {
    if (!message.trim() || contextPapers.length === 0 || chatLoading) return;

    // Add user message immediately
    setChatMessages(prev => [...prev, { role: 'user', content: message }]);
    setChatInput('');
    setChatLoading(true);

    try {
      // Find full paper info from papers array
      const paperContexts = contextPapers.map(cp => {
        const fullPaper = papers.find(p => p.arxiv_id === cp.id);
        return {
          arxiv_id: cp.id,
          title: cp.title,
          abstract: cp.abstract,
          pdf_url: fullPaper?.pdf_url || `https://arxiv.org/pdf/${cp.id}.pdf`,
        };
      });

      const response = await fetch(`${getApiUrl()}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          papers: paperContexts,
          history: chatMessages.slice(-10), // Last 10 messages for context
          parse_pdfs: true,
          use_web_search: webSearchEnabled,
        }),
      });

      if (!response.ok) {
        throw new Error(`Chat failed: ${response.status}`);
      }

      const data = await response.json();
      setChatMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
    } catch (error) {
      console.error('Chat error:', error);
      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, there was an error processing your request. Please try again.'
      }]);
    } finally {
      setChatLoading(false);
    }
  }, [contextPapers, papers, chatMessages, chatLoading, webSearchEnabled]);

  // Bioluminescent node rendering - living neurons
  const nodeCanvasObject = useCallback(
    (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const isHovered = hoveredNode?.id === node.id;
      const isInContext = contextPapers.some(p => p.id === node.id);
      const isBeingDragged = draggingNode?.id === node.id;
      const time = animationFrame / 60;

      // Unique phase per node for organic feel
      const nodePhase = node.id.split('').reduce((a: number, c: string) => a + c.charCodeAt(0), 0) * 0.1;
      const breathe = Math.sin(time * Math.PI + nodePhase) * 0.5 + 0.5;

      // Dynamic sizing: base + citation boost + breathing
      const baseSize = 4;
      const citationBoost = node.pulse_intensity * 6;
      const hoverBoost = isHovered ? 4 : 0;
      const dragBoost = isBeingDragged ? 6 : 0;
      const contextBoost = isInContext ? 2 : 0;
      const breatheBoost = breathe * (1 + node.pulse_intensity * 2);
      const size = baseSize + citationBoost + hoverBoost + dragBoost + contextBoost + breatheBoost;

      // === CONTEXT INDICATOR (cyan ring for papers in dock) ===
      if (isInContext) {
        const contextRingRadius = size * 2;
        // Pulsing context ring
        const contextPhase = (time * 3) % 1;
        ctx.beginPath();
        ctx.arc(node.x, node.y, contextRingRadius * (0.8 + contextPhase * 0.4), 0, Math.PI * 2);
        ctx.strokeStyle = "#00f5d4";
        ctx.lineWidth = 2;
        ctx.globalAlpha = (1 - contextPhase) * 0.8;
        ctx.stroke();
        ctx.globalAlpha = 1;

        // Solid context ring
        ctx.beginPath();
        ctx.arc(node.x, node.y, size * 1.6, 0, Math.PI * 2);
        ctx.strokeStyle = "#00f5d4";
        ctx.lineWidth = 2.5;
        ctx.globalAlpha = 0.9;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      // === DRAG EFFECT (intense glow when being dragged) ===
      if (isBeingDragged) {
        const dragGlowRadius = size * 5;
        const dragGrad = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, dragGlowRadius);
        dragGrad.addColorStop(0, "#00f5d4" + "90");
        dragGrad.addColorStop(0.3, node.color + "60");
        dragGrad.addColorStop(1, "transparent");
        ctx.beginPath();
        ctx.arc(node.x, node.y, dragGlowRadius, 0, Math.PI * 2);
        ctx.fillStyle = dragGrad;
        ctx.fill();
      }

      // === OUTER PLASMA FIELD (for high-citation nodes) ===
      if (node.pulse_intensity > 0.2 || isHovered || isBeingDragged) {
        const plasmaRadius = size * 3;
        const plasmaIntensity = isBeingDragged ? 0.6 : isHovered ? 0.4 : 0.15 + node.pulse_intensity * 0.2;

        // Pulsing plasma ring
        const ringPhase = (time * 2 + nodePhase) % 1;
        const ringRadius = size * (1.5 + ringPhase * 1.5);
        const ringOpacity = (1 - ringPhase) * plasmaIntensity;

        ctx.beginPath();
        ctx.arc(node.x, node.y, ringRadius, 0, Math.PI * 2);
        ctx.strokeStyle = isBeingDragged ? "#00f5d4" : node.color;
        ctx.lineWidth = 2 * (1 - ringPhase);
        ctx.globalAlpha = ringOpacity;
        ctx.stroke();
        ctx.globalAlpha = 1;

        // Diffuse glow
        const glowGrad = ctx.createRadialGradient(node.x, node.y, size * 0.5, node.x, node.y, plasmaRadius);
        glowGrad.addColorStop(0, (isBeingDragged ? "#00f5d4" : node.color) + "60");
        glowGrad.addColorStop(0.4, (isBeingDragged ? "#00f5d4" : node.color) + "20");
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
    [hoveredNode, animationFrame, contextPapers, draggingNode]
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

  // True glassmorphic sidebar - more transparent, ethereal
  const glassSidebar: React.CSSProperties = {
    background: "rgba(10, 10, 18, 0.45)",
    backdropFilter: "blur(30px)",
    WebkitBackdropFilter: "blur(30px)",
    border: "1px solid rgba(255, 255, 255, 0.04)",
    boxShadow: "inset 0 1px 0 0 rgba(255,255,255,0.02), 0 20px 40px -10px rgba(0, 0, 0, 0.4)",
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
    <div style={{ height: "100vh", position: "relative", zIndex: 1, overflow: "hidden" }}>
      {/* Full-screen Main Content */}
      <main style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <div ref={containerRef} style={{ flex: 1, position: "relative", overflow: "hidden" }}>
          {/* ===== SYNAPTIC CONTEXT DOCK ===== */}
          {graphData && (
            <div
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                height: dockZoneHeight,
                zIndex: 20,
                pointerEvents: contextPapers.length > 0 || isNearDock ? "auto" : "none",
              }}
            >
              {/* Receptor membrane background */}
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  background: isNearDock
                    ? "linear-gradient(180deg, rgba(0, 245, 212, 0.15) 0%, rgba(0, 245, 212, 0.05) 60%, transparent 100%)"
                    : contextPapers.length > 0
                    ? "linear-gradient(180deg, rgba(15, 15, 25, 0.9) 0%, rgba(15, 15, 25, 0.7) 60%, transparent 100%)"
                    : "linear-gradient(180deg, rgba(15, 15, 25, 0.3) 0%, transparent 100%)",
                  backdropFilter: contextPapers.length > 0 ? "blur(12px)" : "none",
                  transition: "all 0.3s cubic-bezier(0.16, 1, 0.3, 1)",
                }}
              />

              {/* Glowing edge when near */}
              <div
                style={{
                  position: "absolute",
                  bottom: 0,
                  left: 0,
                  right: 0,
                  height: 2,
                  background: isNearDock
                    ? "linear-gradient(90deg, transparent 0%, #00f5d4 20%, #00f5d4 80%, transparent 100%)"
                    : contextPapers.length > 0
                    ? "linear-gradient(90deg, transparent 0%, rgba(0, 245, 212, 0.3) 20%, rgba(0, 245, 212, 0.3) 80%, transparent 100%)"
                    : "transparent",
                  boxShadow: isNearDock ? "0 0 20px #00f5d4, 0 0 40px rgba(0, 245, 212, 0.5)" : "none",
                  transition: "all 0.3s ease",
                }}
              />

              {/* Reaching tendrils when dragging near */}
              {isNearDock && dragScreenPos && (
                <svg
                  style={{
                    position: "absolute",
                    inset: 0,
                    width: "100%",
                    height: "100%",
                    pointerEvents: "none",
                  }}
                >
                  {[...Array(5)].map((_, i) => {
                    const baseX = dragScreenPos.x;
                    const offsetX = (i - 2) * 30;
                    const waveOffset = Math.sin(animationFrame * 0.1 + i) * 10;
                    return (
                      <path
                        key={i}
                        d={`M ${baseX + offsetX + waveOffset} ${dockZoneHeight}
                            Q ${baseX + offsetX / 2} ${dockZoneHeight * 0.5}
                              ${baseX} ${dragScreenPos.y + 20}`}
                        fill="none"
                        stroke="#00f5d4"
                        strokeWidth={2 - Math.abs(i - 2) * 0.3}
                        strokeOpacity={0.6 - Math.abs(i - 2) * 0.15}
                        style={{
                          filter: "drop-shadow(0 0 4px #00f5d4)",
                        }}
                      />
                    );
                  })}
                </svg>
              )}

              {/* Dock Controls - Right side (just chat button) */}
              <div
                style={{
                  position: "absolute",
                  top: 12,
                  right: 20,
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  zIndex: 25,
                }}
              >
                {/* Chat Button */}
                <button
                  onClick={() => setChatOpen(true)}
                  disabled={contextPapers.length === 0}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    padding: "8px 14px",
                    background: contextPapers.length > 0
                      ? "linear-gradient(135deg, rgba(0, 245, 212, 0.2) 0%, rgba(155, 93, 229, 0.2) 100%)"
                      : "rgba(255, 255, 255, 0.03)",
                    border: "1px solid",
                    borderColor: contextPapers.length > 0 ? "rgba(0, 245, 212, 0.4)" : "rgba(255, 255, 255, 0.06)",
                    borderRadius: "8px",
                    color: contextPapers.length > 0 ? "#00f5d4" : "var(--text-muted)",
                    fontSize: "0.8rem",
                    fontWeight: 500,
                    cursor: contextPapers.length > 0 ? "pointer" : "default",
                    transition: "all 0.2s ease",
                    boxShadow: contextPapers.length > 0 ? "0 0 20px rgba(0, 245, 212, 0.15)" : "none",
                  }}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                  </svg>
                  Chat
                  {contextPapers.length > 0 && (
                    <span
                      style={{
                        padding: "2px 6px",
                        background: "#00f5d4",
                        borderRadius: "100px",
                        fontSize: "0.6rem",
                        color: "#000",
                        fontWeight: 600,
                      }}
                    >
                      {contextPapers.length}
                    </span>
                  )}
                </button>
              </div>

              {/* Dock label - Left side */}
              <div
                style={{
                  position: "absolute",
                  top: 12,
                  left: 20,
                  fontSize: "0.65rem",
                  fontFamily: "var(--font-mono)",
                  textTransform: "uppercase",
                  letterSpacing: "0.15em",
                  color: isNearDock ? "#00f5d4" : "var(--text-muted)",
                  transition: "color 0.3s ease",
                  opacity: contextPapers.length === 0 && !isNearDock ? 0.5 : 1,
                }}
              >
                {isNearDock ? "Release to add context" : contextPapers.length > 0 ? `${contextPapers.length} paper${contextPapers.length !== 1 ? 's' : ''} in context` : "Drag papers here"}
              </div>

              {/* Docked papers - neural circuit */}
              {contextPapers.length > 0 && (
                <div
                  style={{
                    position: "absolute",
                    top: 32,
                    left: 24,
                    right: 24,
                    display: "flex",
                    alignItems: "center",
                    gap: 0,
                    justifyContent: "center",
                  }}
                >
                  {/* Axon line connecting all papers */}
                  <svg
                    style={{
                      position: "absolute",
                      top: "50%",
                      left: 0,
                      right: 0,
                      height: 4,
                      transform: "translateY(-50%)",
                      overflow: "visible",
                    }}
                  >
                    <defs>
                      <linearGradient id="axonGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="transparent" />
                        <stop offset="10%" stopColor="#00f5d4" stopOpacity="0.4" />
                        <stop offset="90%" stopColor="#00f5d4" stopOpacity="0.4" />
                        <stop offset="100%" stopColor="transparent" />
                      </linearGradient>
                      <filter id="axonGlow">
                        <feGaussianBlur stdDeviation="2" result="blur" />
                        <feMerge>
                          <feMergeNode in="blur" />
                          <feMergeNode in="SourceGraphic" />
                        </feMerge>
                      </filter>
                    </defs>
                    <line
                      x1="10%"
                      y1="2"
                      x2="90%"
                      y2="2"
                      stroke="url(#axonGradient)"
                      strokeWidth="2"
                      filter="url(#axonGlow)"
                    />
                    {/* Traveling pulse along axon */}
                    <circle
                      cx={`${10 + ((animationFrame * 0.5) % 80)}%`}
                      cy="2"
                      r="4"
                      fill="#00f5d4"
                      filter="url(#axonGlow)"
                    >
                      <animate
                        attributeName="opacity"
                        values="1;0.5;1"
                        dur="1s"
                        repeatCount="indefinite"
                      />
                    </circle>
                  </svg>

                  {/* Paper nodes */}
                  {contextPapers.map((paper, index) => (
                    <div
                      key={paper.id}
                      style={{
                        position: "relative",
                        display: "flex",
                        alignItems: "center",
                        padding: "0 16px",
                        animation: "fade-in-up 0.3s ease-out",
                        animationDelay: `${index * 0.05}s`,
                        animationFillMode: "both",
                      }}
                    >
                      {/* Mini neuron */}
                      <div
                        style={{
                          position: "relative",
                          width: 44,
                          height: 44,
                          borderRadius: "50%",
                          background: `radial-gradient(circle at 30% 30%, ${paper.color}, ${paper.color}80)`,
                          boxShadow: `0 0 15px ${paper.color}60, inset 0 0 10px rgba(255,255,255,0.2)`,
                          cursor: "pointer",
                          transition: "transform 0.2s ease, box-shadow 0.2s ease",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                        onMouseEnter={(e) => {
                          (e.target as HTMLElement).style.transform = "scale(1.1)";
                          (e.target as HTMLElement).style.boxShadow = `0 0 25px ${paper.color}80, inset 0 0 10px rgba(255,255,255,0.3)`;
                        }}
                        onMouseLeave={(e) => {
                          (e.target as HTMLElement).style.transform = "scale(1)";
                          (e.target as HTMLElement).style.boxShadow = `0 0 15px ${paper.color}60, inset 0 0 10px rgba(255,255,255,0.2)`;
                        }}
                        title={paper.title}
                      >
                        {/* Specular highlight */}
                        <div
                          style={{
                            position: "absolute",
                            top: 6,
                            left: 10,
                            width: 12,
                            height: 8,
                            borderRadius: "50%",
                            background: "rgba(255,255,255,0.5)",
                          }}
                        />
                        {/* Parsed status badge */}
                        {parsingPapers.has(paper.id) ? (
                          <div
                            style={{
                              position: "absolute",
                              bottom: -4,
                              left: "50%",
                              transform: "translateX(-50%)",
                              width: 18,
                              height: 18,
                              borderRadius: "50%",
                              background: "rgba(0, 180, 216, 0.9)",
                              border: "2px solid rgba(10, 10, 18, 0.9)",
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              boxShadow: "0 0 8px rgba(0, 180, 216, 0.6)",
                            }}
                          >
                            <div
                              style={{
                                width: 10,
                                height: 10,
                                border: "2px solid transparent",
                                borderTopColor: "#fff",
                                borderRadius: "50%",
                                animation: "spin 0.8s linear infinite",
                              }}
                            />
                          </div>
                        ) : parsedPapers.has(paper.id) ? (
                          <div
                            style={{
                              position: "absolute",
                              bottom: -4,
                              left: "50%",
                              transform: "translateX(-50%)",
                              width: 18,
                              height: 18,
                              borderRadius: "50%",
                              background: "rgba(0, 245, 212, 0.9)",
                              border: "2px solid rgba(10, 10, 18, 0.9)",
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              boxShadow: "0 0 8px rgba(0, 245, 212, 0.6)",
                            }}
                            title="PDF parsed"
                          >
                            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#000" strokeWidth="4">
                              <path d="M20 6L9 17l-5-5" />
                            </svg>
                          </div>
                        ) : null}
                        {/* Remove button */}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            removeFromContext(paper.id);
                          }}
                          style={{
                            position: "absolute",
                            top: -4,
                            right: -4,
                            width: 18,
                            height: 18,
                            borderRadius: "50%",
                            background: "rgba(241, 91, 181, 0.9)",
                            border: "none",
                            cursor: "pointer",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            fontSize: "10px",
                            color: "#fff",
                            fontWeight: "bold",
                            boxShadow: "0 0 8px rgba(241, 91, 181, 0.6)",
                            transition: "transform 0.15s ease",
                          }}
                          onMouseEnter={(e) => {
                            (e.target as HTMLElement).style.transform = "scale(1.2)";
                          }}
                          onMouseLeave={(e) => {
                            (e.target as HTMLElement).style.transform = "scale(1)";
                          }}
                        >
                          ×
                        </button>
                      </div>

                      {/* Paper title tooltip on hover (appears below) */}
                      <div
                        style={{
                          position: "absolute",
                          top: "100%",
                          left: "50%",
                          transform: "translateX(-50%)",
                          marginTop: 8,
                          padding: "6px 10px",
                          background: "rgba(15, 15, 25, 0.95)",
                          border: "1px solid rgba(255,255,255,0.1)",
                          borderRadius: 6,
                          fontSize: "0.7rem",
                          color: "var(--text-secondary)",
                          whiteSpace: "nowrap",
                          maxWidth: 200,
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          opacity: 0,
                          pointerEvents: "none",
                          transition: "opacity 0.2s ease",
                          zIndex: 30,
                        }}
                        className="paper-tooltip"
                      >
                        {paper.title.slice(0, 50)}{paper.title.length > 50 ? '...' : ''}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Empty state hint */}
              {contextPapers.length === 0 && !isNearDock && graphData && (
                <div
                  style={{
                    position: "absolute",
                    top: "50%",
                    left: "50%",
                    transform: "translate(-50%, -50%)",
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    fontSize: "0.75rem",
                    color: "var(--text-muted)",
                    opacity: 0.6,
                  }}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M12 5v14M5 12h14" />
                  </svg>
                  Drag papers here to build context
                </div>
              )}
            </div>
          )}
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
              // Drag handler for synaptic dock
              onNodeDrag={(node: any) => {
                const graphNode = node as GraphNode;
                draggingNodeRef.current = graphNode;
                setDraggingNode(graphNode);

                // Get screen position
                if (graphRef.current) {
                  const coords = graphRef.current.graph2ScreenCoords(node.x, node.y);
                  if (coords) {
                    const nearDock = coords.y < dockZoneHeight + 50;
                    setDragScreenPos({ x: coords.x, y: coords.y });
                    setIsNearDock(nearDock);
                    isNearDockRef.current = nearDock;
                  }
                }
              }}
              {...{
                onNodeDragEnd: (node: any) => {
                  const graphNode = node as GraphNode;

                  // Check if in dock zone - add to context
                  if (graphRef.current) {
                    const coords = graphRef.current.graph2ScreenCoords(node.x, node.y);
                    if (coords && coords.y < dockZoneHeight + 50) {
                      setContextPapers(prev => {
                        if (prev.find(p => p.id === graphNode.id)) return prev;
                        return [...prev, graphNode];
                      });
                    }
                  }

                  // Reset drag state
                  setDraggingNode(null);
                  setDragScreenPos(null);
                  setIsNearDock(false);
                  draggingNodeRef.current = null;
                  isNearDockRef.current = false;
                }
              } as any}
              // Force simulation for organic node spacing
              d3AlphaDecay={0.015}
              d3VelocityDecay={0.25}
              cooldownTime={4000}
              warmupTicks={150}
              enableNodeDrag={true}
              backgroundColor="transparent"
            />
          )}

          {/* Tooltip - follows node position */}
          {hoveredNode && tooltipPos && (() => {
            // Get ALL connected nodes from graph links
            const connectedSimilar: GraphNode[] = [];
            const connectedCitations: GraphNode[] = [];
            const seenIds = new Set<string>();

            filteredGraphData?.links.forEach(link => {
              const sourceId = typeof link.source === 'string' ? link.source : (link.source as GraphNode).id;
              const targetId = typeof link.target === 'string' ? link.target : (link.target as GraphNode).id;

              let connectedId: string | null = null;
              if (sourceId === hoveredNode.id) connectedId = targetId;
              else if (targetId === hoveredNode.id) connectedId = sourceId;

              if (connectedId && !seenIds.has(connectedId)) {
                seenIds.add(connectedId);
                const node = filteredGraphData?.nodes.find(n => n.id === connectedId);
                if (node) {
                  if (link.isCitation) {
                    connectedCitations.push(node);
                  } else {
                    connectedSimilar.push(node);
                  }
                }
              }
            });

            const hasConnections = connectedSimilar.length > 0 || connectedCitations.length > 0;

            // Smart positioning - prefer right side, flip if near edge
            const tooltipWidth = 340;
            const totalConnections = connectedSimilar.length + connectedCitations.length;
            const tooltipHeight = Math.min(400, 180 + totalConnections * 20); // Dynamic height
            const padding = 20;
            const nodeOffset = 30;

            // Calculate position - prefer to the right of the node
            let left = tooltipPos.x + nodeOffset;
            let top = tooltipPos.y - tooltipHeight / 2;

            // Flip to left if would overflow right edge
            if (left + tooltipWidth > dimensions.width - padding) {
              left = tooltipPos.x - tooltipWidth - nodeOffset;
            }

            // Keep within vertical bounds
            if (top < dockZoneHeight + padding) {
              top = dockZoneHeight + padding;
            }
            if (top + tooltipHeight > dimensions.height - padding) {
              top = dimensions.height - tooltipHeight - padding;
            }

            // Don't show if would overlap with left panel
            const leftPanelRight = sidebarCollapsed ? 80 : 360;
            if (left < leftPanelRight && tooltipPos.x < leftPanelRight + 100) {
              left = leftPanelRight + padding;
            }

            return (
              <div
                style={{
                  ...glassPanel,
                  position: "absolute",
                  left,
                  top,
                  width: tooltipWidth,
                  maxHeight: 400,
                  overflow: "auto",
                  padding: "16px 18px",
                  borderRadius: "12px",
                  zIndex: 25,
                  pointerEvents: "none",
                  animation: "fade-in-up 0.15s ease-out",
                }}
              >
                {/* Main paper info */}
                <div style={{ display: "flex", gap: "12px", marginBottom: hasConnections ? 8 : 0 }}>
                  <div
                    style={{
                      width: 10,
                      height: 10,
                      borderRadius: "50%",
                      background: hoveredNode.color,
                      marginTop: 4,
                      flexShrink: 0,
                      boxShadow: `0 0 10px ${hoveredNode.color}60`,
                    }}
                  />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <h3 style={{
                      fontSize: "0.85rem",
                      fontWeight: 600,
                      lineHeight: 1.35,
                      marginBottom: 4,
                    }}>
                      {hoveredNode.title.length > 80 ? hoveredNode.title.slice(0, 80) + '...' : hoveredNode.title}
                    </h3>
                    <p style={{
                      fontSize: "0.65rem",
                      color: "var(--text-secondary)",
                      marginBottom: 6,
                      fontFamily: "var(--font-mono)",
                      display: "flex",
                      flexWrap: "wrap",
                      gap: "4px",
                      alignItems: "center",
                    }}>
                      <span>{hoveredNode.authors.slice(0, 2).join(", ")}{hoveredNode.authors.length > 2 && ` +${hoveredNode.authors.length - 2}`}</span>
                      <span style={{ color: "var(--text-muted)" }}>·</span>
                      <span>{new Date(hoveredNode.published).toLocaleDateString()}</span>
                      {hoveredNode.citation_count > 0 && (
                        <>
                          <span style={{ color: "var(--text-muted)" }}>·</span>
                          <span style={{ color: "#00f5d4" }}>{hoveredNode.citation_count} cites</span>
                        </>
                      )}
                    </p>
                    <p style={{
                      fontSize: "0.7rem",
                      color: "var(--text-tertiary)",
                      lineHeight: 1.5,
                      display: "-webkit-box",
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: "vertical",
                      overflow: "hidden",
                    }}>
                      {hoveredNode.abstract}
                    </p>
                  </div>
                </div>

                {/* Connected papers - Similar */}
                {connectedSimilar.length > 0 && (
                  <div
                    style={{
                      borderTop: "1px solid rgba(255, 255, 255, 0.06)",
                      paddingTop: 10,
                      marginTop: 4,
                    }}
                  >
                    <div style={{
                      fontSize: "0.55rem",
                      color: "var(--text-muted)",
                      textTransform: "uppercase",
                      letterSpacing: "0.1em",
                      marginBottom: 6,
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                    }}>
                      <span style={{ color: "var(--text-tertiary)" }}>Similar</span>
                      <span style={{
                        background: "rgba(155, 93, 229, 0.3)",
                        padding: "1px 5px",
                        borderRadius: "4px",
                        fontSize: "0.5rem",
                      }}>
                        {connectedSimilar.length}
                      </span>
                    </div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                      {connectedSimilar.map(node => (
                        <div
                          key={node.id}
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 4,
                            padding: "3px 8px",
                            background: "rgba(255, 255, 255, 0.03)",
                            borderRadius: "4px",
                            maxWidth: "100%",
                          }}
                          title={node.title}
                        >
                          <div
                            style={{
                              width: 5,
                              height: 5,
                              borderRadius: "50%",
                              background: node.color,
                              flexShrink: 0,
                              boxShadow: `0 0 4px ${node.color}40`,
                            }}
                          />
                          <span style={{
                            fontSize: "0.55rem",
                            color: "var(--text-secondary)",
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                            maxWidth: 140,
                          }}>
                            {node.title.length > 25 ? node.title.slice(0, 25) + '...' : node.title}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Connected papers - Citations */}
                {connectedCitations.length > 0 && (
                  <div
                    style={{
                      borderTop: "1px solid rgba(255, 255, 255, 0.06)",
                      paddingTop: 10,
                      marginTop: connectedSimilar.length > 0 ? 8 : 4,
                    }}
                  >
                    <div style={{
                      fontSize: "0.55rem",
                      color: "var(--text-muted)",
                      textTransform: "uppercase",
                      letterSpacing: "0.1em",
                      marginBottom: 6,
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                    }}>
                      <span style={{ color: "#00f5d4" }}>Citations</span>
                      <span style={{
                        background: "rgba(0, 245, 212, 0.2)",
                        padding: "1px 5px",
                        borderRadius: "4px",
                        fontSize: "0.5rem",
                        color: "#00f5d4",
                      }}>
                        {connectedCitations.length}
                      </span>
                    </div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                      {connectedCitations.map(node => (
                        <div
                          key={node.id}
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 4,
                            padding: "3px 8px",
                            background: "rgba(0, 245, 212, 0.05)",
                            border: "1px solid rgba(0, 245, 212, 0.15)",
                            borderRadius: "4px",
                            maxWidth: "100%",
                          }}
                          title={node.title}
                        >
                          <div
                            style={{
                              width: 5,
                              height: 5,
                              borderRadius: "50%",
                              background: "#00f5d4",
                              flexShrink: 0,
                              boxShadow: "0 0 4px rgba(0, 245, 212, 0.4)",
                            }}
                          />
                          <span style={{
                            fontSize: "0.55rem",
                            color: "var(--text-secondary)",
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                            maxWidth: 140,
                          }}>
                            {node.title.length > 25 ? node.title.slice(0, 25) + '...' : node.title}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })()}

          {/* ===== FLOATING GLASS PANEL ===== */}
          <div
            style={{
              position: "absolute",
              top: "50%",
              left: 24,
              transform: "translateY(-50%)",
              zIndex: 30,
              display: "flex",
              flexDirection: "column",
              maxWidth: sidebarCollapsed ? 56 : 320,
              maxHeight: "calc(100% - 140px)",
              background: "rgba(10, 10, 18, 0.6)",
              backdropFilter: "blur(24px)",
              WebkitBackdropFilter: "blur(24px)",
              border: "1px solid rgba(255, 255, 255, 0.08)",
              borderRadius: "16px",
              boxShadow: "0 20px 50px -15px rgba(0, 0, 0, 0.5), inset 0 1px 0 0 rgba(255,255,255,0.05)",
              overflow: "hidden",
              transition: "all 0.3s cubic-bezier(0.16, 1, 0.3, 1)",
            }}
          >
            {/* Header with collapse toggle */}
            <div
              style={{
                padding: sidebarCollapsed ? "12px" : "16px 20px",
                borderBottom: sidebarCollapsed ? "none" : "1px solid rgba(255, 255, 255, 0.04)",
                display: "flex",
                alignItems: "center",
                justifyContent: sidebarCollapsed ? "center" : "space-between",
                gap: 12,
              }}
            >
              {sidebarCollapsed ? (
                <button
                  onClick={() => setSidebarCollapsed(false)}
                  style={{
                    width: 32,
                    height: 32,
                    borderRadius: "8px",
                    background: "linear-gradient(135deg, #00f5d4 0%, #9b5de5 100%)",
                    border: "none",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontWeight: 700,
                    fontSize: "1rem",
                    color: "#000",
                  }}
                >
                  P
                </button>
              ) : (
                <>
                  <div>
                    <h1
                      style={{
                        fontSize: "1.1rem",
                        fontWeight: 600,
                        background: "linear-gradient(135deg, #00f5d4 0%, #9b5de5 100%)",
                        WebkitBackgroundClip: "text",
                        WebkitTextFillColor: "transparent",
                        letterSpacing: "-0.02em",
                      }}
                    >
                      PaperMap
                    </h1>
                  </div>
                  <button
                    onClick={() => setSidebarCollapsed(true)}
                    style={{
                      width: 24,
                      height: 24,
                      borderRadius: "6px",
                      background: "rgba(255, 255, 255, 0.05)",
                      border: "1px solid rgba(255, 255, 255, 0.08)",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--text-secondary)" strokeWidth="2">
                      <path d="M18 6L6 18M6 6l12 12" />
                    </svg>
                  </button>
                </>
              )}
            </div>

            {/* Expanded content */}
            {!sidebarCollapsed && (
              <>
                {/* Search */}
                <div style={{ padding: "16px 20px", borderBottom: "1px solid rgba(255, 255, 255, 0.04)" }}>
                  <div style={{ display: "flex", gap: "8px" }}>
                    <input
                      type="text"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                      placeholder="Search papers..."
                      disabled={isLoading}
                      style={{
                        ...inputStyle,
                        padding: "10px 12px",
                        fontSize: "0.85rem",
                        background: "rgba(255, 255, 255, 0.03)",
                      }}
                    />
                    <button
                      onClick={handleSearch}
                      disabled={isLoading || !query.trim()}
                      style={{
                        padding: "10px 14px",
                        background: isLoading
                          ? "rgba(255, 255, 255, 0.05)"
                          : "linear-gradient(135deg, #00f5d4 0%, #00b4d8 100%)",
                        border: "none",
                        borderRadius: "8px",
                        color: isLoading ? "var(--text-muted)" : "#000",
                        fontSize: "0.8rem",
                        fontWeight: 600,
                        cursor: isLoading ? "not-allowed" : "pointer",
                        opacity: isLoading || !query.trim() ? 0.5 : 1,
                        transition: "all 0.2s ease",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {isLoading ? "..." : "Go"}
                    </button>
                  </div>
                  <select
                    value={paperCount}
                    onChange={(e) => setPaperCount(Number(e.target.value))}
                    disabled={isLoading}
                    style={{
                      width: "100%",
                      marginTop: 8,
                      padding: "8px 10px",
                      background: "rgba(255, 255, 255, 0.03)",
                      border: "1px solid rgba(255, 255, 255, 0.06)",
                      borderRadius: "6px",
                      color: "var(--text-secondary)",
                      fontSize: "0.75rem",
                      cursor: "pointer",
                      appearance: "none",
                    }}
                  >
                    {PAPER_COUNTS.map((n) => (
                      <option key={n} value={n}>{n} papers</option>
                    ))}
                  </select>
                </div>

                {/* Scrollable content */}
                <div style={{ flex: 1, overflow: "auto", padding: "16px 20px" }}>
                  {/* Expanded Queries */}
                  {expandedQueries.length > 0 && (
                    <div style={{ marginBottom: "16px" }}>
                      <div
                        style={{
                          fontSize: "0.6rem",
                          color: "var(--text-muted)",
                          textTransform: "uppercase",
                          letterSpacing: "0.1em",
                          marginBottom: "8px",
                          fontWeight: 500,
                        }}
                      >
                        Expanded
                      </div>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
                        {expandedQueries.slice(0, 4).map((q, i) => (
                          <span
                            key={i}
                            style={{
                              fontSize: "0.6rem",
                              padding: "3px 6px",
                              background: "rgba(0, 245, 212, 0.08)",
                              border: "1px solid rgba(0, 245, 212, 0.15)",
                              borderRadius: "4px",
                              color: "var(--text-secondary)",
                              fontFamily: "var(--font-mono)",
                            }}
                          >
                            {q.length > 20 ? q.slice(0, 20) + "..." : q}
                          </span>
                        ))}
                        {expandedQueries.length > 4 && (
                          <span style={{ fontSize: "0.6rem", color: "var(--text-muted)", padding: "3px 6px" }}>
                            +{expandedQueries.length - 4}
                          </span>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Filters */}
                  {categories.length > 0 && (
                    <div style={{ marginBottom: "16px" }}>
                      <div
                        style={{
                          fontSize: "0.6rem",
                          color: "var(--text-muted)",
                          textTransform: "uppercase",
                          letterSpacing: "0.1em",
                          marginBottom: "8px",
                          fontWeight: 500,
                        }}
                      >
                        Time Filter
                      </div>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
                        {[
                          { value: "all", label: "All" },
                          { value: "5y", label: "5y" },
                          { value: "2y", label: "2y" },
                          { value: "1y", label: "1y" },
                        ].map((opt) => (
                          <button
                            key={opt.value}
                            onClick={() => setDateFilter(opt.value)}
                            style={{
                              padding: "4px 10px",
                              borderRadius: "6px",
                              fontSize: "0.7rem",
                              border: "1px solid",
                              borderColor: dateFilter === opt.value ? "#00f5d4" : "rgba(255, 255, 255, 0.08)",
                              background: dateFilter === opt.value ? "rgba(0, 245, 212, 0.15)" : "transparent",
                              color: dateFilter === opt.value ? "#00f5d4" : "var(--text-secondary)",
                              cursor: "pointer",
                              transition: "all 0.15s ease",
                            }}
                          >
                            {opt.label}
                          </button>
                        ))}
                      </div>
                      <div
                        style={{
                          fontSize: "0.6rem",
                          color: "var(--text-muted)",
                          marginTop: "8px",
                          fontFamily: "var(--font-mono)",
                        }}
                      >
                        {visibleCount}/{totalCount} visible
                      </div>
                    </div>
                  )}

                  {/* Categories */}
                  {categories.length > 0 && (
                    <div>
                      <div
                        style={{
                          fontSize: "0.6rem",
                          color: "var(--text-muted)",
                          textTransform: "uppercase",
                          letterSpacing: "0.1em",
                          marginBottom: "8px",
                          fontWeight: 500,
                        }}
                      >
                        Clusters
                      </div>
                      <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                        {categories.map((cat) => (
                          <button
                            key={cat.id}
                            onClick={() => {
                              const newSet = new Set(visibleClusters);
                              if (newSet.has(cat.id)) {
                                newSet.delete(cat.id);
                              } else {
                                newSet.add(cat.id);
                              }
                              setVisibleClusters(newSet);
                            }}
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: "8px",
                              padding: "6px 8px",
                              background: visibleClusters.has(cat.id) ? "rgba(255, 255, 255, 0.03)" : "transparent",
                              borderRadius: "6px",
                              border: "none",
                              cursor: "pointer",
                              opacity: visibleClusters.has(cat.id) ? 1 : 0.5,
                              transition: "all 0.15s ease",
                              textAlign: "left",
                            }}
                            title={cat.description}
                          >
                            <span
                              style={{
                                width: 10,
                                height: 10,
                                borderRadius: "3px",
                                background: visibleClusters.has(cat.id) ? cat.color : "transparent",
                                border: `2px solid ${cat.color}`,
                                flexShrink: 0,
                                boxShadow: visibleClusters.has(cat.id) ? `0 0 6px ${cat.color}40` : "none",
                              }}
                            />
                            <span style={{
                              flex: 1,
                              fontSize: "0.7rem",
                              color: visibleClusters.has(cat.id) ? "var(--text-primary)" : "var(--text-muted)",
                              fontWeight: 500,
                              whiteSpace: "nowrap",
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                            }}>
                              {cat.name}
                            </span>
                            <span style={{
                              fontSize: "0.6rem",
                              color: "var(--text-muted)",
                              fontFamily: "var(--font-mono)",
                            }}>
                              {cat.count}
                            </span>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Empty state */}
                  {categories.length === 0 && !isLoading && (
                    <div style={{ textAlign: "center", padding: "20px 0", color: "var(--text-muted)" }}>
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" style={{ margin: "0 auto 8px", opacity: 0.5 }}>
                        <circle cx="11" cy="11" r="8" />
                        <path d="M21 21l-4.35-4.35" />
                      </svg>
                      <p style={{ fontSize: "0.75rem" }}>Search to explore</p>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      </main>

      {/* Chat Panel - Floating overlay from right */}
      {chatOpen && (
        <aside
          style={{
            position: "absolute",
            top: 24,
            right: 24,
            bottom: 24,
            width: 380,
            maxWidth: "calc(100% - 48px)",
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
            background: "rgba(10, 10, 18, 0.85)",
            backdropFilter: "blur(24px)",
            WebkitBackdropFilter: "blur(24px)",
            border: "1px solid rgba(255, 255, 255, 0.08)",
            borderRadius: "16px",
            boxShadow: "0 20px 50px -15px rgba(0, 0, 0, 0.5), inset 0 1px 0 0 rgba(255,255,255,0.05)",
            animation: "slide-in-right 0.3s cubic-bezier(0.16, 1, 0.3, 1)",
            zIndex: 40,
          }}
        >
          {/* Chat Header */}
          <div
            style={{
              padding: "16px 20px",
              borderBottom: "1px solid rgba(255, 255, 255, 0.04)",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: "50%",
                  background: "#00f5d4",
                  boxShadow: "0 0 10px #00f5d460",
                  animation: "glow-pulse 2s ease-in-out infinite",
                }}
              />
              <span style={{ fontSize: "0.9rem", fontWeight: 600 }}>
                Research Chat
              </span>
              <span
                style={{
                  fontSize: "0.65rem",
                  color: "var(--text-muted)",
                  fontFamily: "var(--font-mono)",
                }}
              >
                {contextPapers.length} papers loaded
              </span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              {/* Web Search Toggle */}
              <button
                onClick={() => setWebSearchEnabled(!webSearchEnabled)}
                title={webSearchEnabled ? "Web search ON - click to disable" : "Web search OFF - click to enable"}
                style={{
                  padding: "4px 10px",
                  borderRadius: "6px",
                  background: webSearchEnabled ? "rgba(0, 245, 212, 0.15)" : "rgba(255, 255, 255, 0.05)",
                  border: `1px solid ${webSearchEnabled ? "rgba(0, 245, 212, 0.3)" : "rgba(255, 255, 255, 0.08)"}`,
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: 5,
                  fontSize: "0.6rem",
                  color: webSearchEnabled ? "#00f5d4" : "var(--text-muted)",
                  fontFamily: "var(--font-mono)",
                  transition: "all 0.2s ease",
                }}
              >
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="11" cy="11" r="8" />
                  <path d="M21 21l-4.35-4.35" />
                </svg>
                Web
              </button>
              {/* Close Button */}
              <button
                onClick={() => setChatOpen(false)}
                style={{
                  width: 28,
                  height: 28,
                  borderRadius: "6px",
                  background: "rgba(255, 255, 255, 0.05)",
                  border: "1px solid rgba(255, 255, 255, 0.08)",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--text-secondary)" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>

          {/* Chat Messages */}
          <div
            style={{
              flex: 1,
              overflow: "auto",
              padding: "20px",
              display: "flex",
              flexDirection: "column",
              gap: "16px",
            }}
          >
            {chatMessages.length === 0 ? (
              <div
                style={{
                  flex: 1,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 12,
                  color: "var(--text-muted)",
                  textAlign: "center",
                  padding: "40px 20px",
                }}
              >
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
                <p style={{ fontSize: "0.85rem" }}>
                  Ask questions about your selected papers
                </p>
                <p style={{ fontSize: "0.7rem", fontFamily: "var(--font-mono)" }}>
                  VLM will parse PDFs for deep understanding
                </p>
              </div>
            ) : (
              <>
                {chatMessages.map((msg, i) => (
                  <div
                    key={i}
                    style={{
                      display: "flex",
                      justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
                    }}
                  >
                    <div
                      style={{
                        maxWidth: "85%",
                        padding: "12px 16px",
                        borderRadius: msg.role === "user" ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
                        background: msg.role === "user"
                          ? "linear-gradient(135deg, rgba(0, 245, 212, 0.2) 0%, rgba(0, 180, 216, 0.2) 100%)"
                          : "rgba(255, 255, 255, 0.05)",
                        border: "1px solid",
                        borderColor: msg.role === "user"
                          ? "rgba(0, 245, 212, 0.3)"
                          : "rgba(255, 255, 255, 0.06)",
                        fontSize: "0.85rem",
                        lineHeight: 1.6,
                      }}
                      className={msg.role === "assistant" ? "chat-markdown" : ""}
                    >
                      {msg.role === "assistant" ? (
                        <Markdown>{msg.content}</Markdown>
                      ) : (
                        msg.content
                      )}
                    </div>
                  </div>
                ))}
                {chatLoading && (
                  <div style={{ display: "flex", justifyContent: "flex-start" }}>
                    <div
                      style={{
                        padding: "12px 16px",
                        borderRadius: "16px 16px 16px 4px",
                        background: "rgba(255, 255, 255, 0.05)",
                        border: "1px solid rgba(255, 255, 255, 0.06)",
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                      }}
                    >
                      <div style={{
                        width: 8,
                        height: 8,
                        borderRadius: "50%",
                        background: "#00f5d4",
                        animation: "glow-pulse 1s ease-in-out infinite",
                      }} />
                      <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                        Analyzing papers...
                      </span>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Chat Input */}
          <div
            style={{
              padding: "16px 20px",
              borderTop: "1px solid rgba(255, 255, 255, 0.04)",
            }}
          >
            <div
              style={{
                display: "flex",
                gap: 10,
                background: "rgba(22, 22, 34, 0.6)",
                borderRadius: "12px",
                border: "1px solid rgba(255, 255, 255, 0.06)",
                padding: "4px",
              }}
            >
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && chatInput.trim() && !chatLoading) {
                    sendChatMessage(chatInput);
                  }
                }}
                placeholder={chatLoading ? "Thinking..." : "Ask about these papers..."}
                disabled={chatLoading}
                style={{
                  flex: 1,
                  padding: "10px 14px",
                  background: "transparent",
                  border: "none",
                  color: "var(--text-primary)",
                  fontSize: "0.85rem",
                  outline: "none",
                  opacity: chatLoading ? 0.6 : 1,
                }}
              />
              <button
                onClick={() => sendChatMessage(chatInput)}
                disabled={!chatInput.trim() || chatLoading}
                style={{
                  padding: "10px 16px",
                  background: chatInput.trim() && !chatLoading
                    ? "linear-gradient(135deg, #00f5d4 0%, #00b4d8 100%)"
                    : "rgba(255, 255, 255, 0.05)",
                  border: "none",
                  borderRadius: "8px",
                  color: chatInput.trim() && !chatLoading ? "#000" : "var(--text-muted)",
                  fontSize: "0.85rem",
                  fontWeight: 600,
                  cursor: chatInput.trim() && !chatLoading ? "pointer" : "default",
                  transition: "all 0.2s ease",
                  minWidth: 60,
                }}
              >
                {chatLoading ? (
                  <div style={{
                    width: 16,
                    height: 16,
                    border: "2px solid rgba(0, 245, 212, 0.3)",
                    borderTopColor: "#00f5d4",
                    borderRadius: "50%",
                    animation: "spin 1s linear infinite",
                    margin: "0 auto",
                  }} />
                ) : "Send"}
              </button>
            </div>
          </div>
        </aside>
      )}
    </div>
  );
}
