"use client";

import { useEffect, useRef } from "react";
import * as d3 from "d3";
import type { VizKGNode, VizKGLink } from "@/lib/api";

interface KGGraphProps {
  nodes: VizKGNode[];
  links: VizKGLink[];
  centerId?: string | null;
  height?: number;
}

interface SimNode extends VizKGNode, d3.SimulationNodeDatum {}
interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  relation: string;
}

const TYPE_COLORS: Record<string, string> = {
  compound: "#8ecae6",
  disease: "#ffb4a2",
  gene: "#b5e48c",
  anatomy: "#ffd166",
  biologicalprocess: "#c8b6ff",
  molecularfunction: "#ffc6ff",
  cellularcomponent: "#9bf6ff",
  pathway: "#fdffb6",
  symptom: "#ff9aa2",
  sideeffect: "#ffafcc",
  pharmacologicclass: "#bdb2ff",
};

function colorFor(kind: string): string {
  const k = (kind || "").toLowerCase().replace(/[\s_]/g, "");
  return TYPE_COLORS[k] ?? "#cdd6e4";
}

export function KGGraph({ nodes, links, centerId, height = 520 }: KGGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;
    if (nodes.length === 0) return;

    const container = containerRef.current;
    const width = container.clientWidth || 800;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const simNodes: SimNode[] = nodes.map((n) => ({ ...n }));
    const idToNode = new Map(simNodes.map((n) => [n.id, n] as const));
    const simLinks: SimLink[] = links
      .filter((l) => idToNode.has(l.source) && idToNode.has(l.target))
      .map((l) => ({
        source: idToNode.get(l.source)!,
        target: idToNode.get(l.target)!,
        relation: l.relation,
      }));

    const zoomGroup = svg.append("g");
    svg.call(
      d3
        .zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.2, 4])
        .on("zoom", (event) => {
          zoomGroup.attr("transform", event.transform.toString());
        }),
    );

    const linkGroup = zoomGroup
      .append("g")
      .attr("stroke", "#5f6b80")
      .attr("stroke-opacity", 0.5);

    const link = linkGroup
      .selectAll("line")
      .data(simLinks)
      .join("line")
      .attr("stroke-width", 1.2);

    link.append("title").text((d) => d.relation);

    const nodeGroup = zoomGroup.append("g");

    const node = nodeGroup
      .selectAll("g")
      .data(simNodes)
      .join("g")
      .attr("cursor", "grab");

    node
      .append("circle")
      .attr("r", (d) => (d.id === centerId ? 10 : 6))
      .attr("fill", (d) => colorFor(d.entity_type))
      .attr("stroke", (d) => (d.id === centerId ? "#ffffff" : "#1b1f2a"))
      .attr("stroke-width", (d) => (d.id === centerId ? 2 : 1));

    node
      .append("title")
      .text((d) => `${d.label}\n${d.id}\n${d.entity_type}`);

    node
      .append("text")
      .text((d) => d.label)
      .attr("x", 10)
      .attr("y", 4)
      .attr("font-size", 10)
      .attr("fill", "#e2e6f0")
      .attr("pointer-events", "none")
      .attr("paint-order", "stroke")
      .attr("stroke", "#1b1f2a")
      .attr("stroke-width", 2.5)
      .attr("stroke-opacity", 0.8);

    const simulation = d3
      .forceSimulation<SimNode>(simNodes)
      .force(
        "link",
        d3
          .forceLink<SimNode, SimLink>(simLinks)
          .id((d) => d.id)
          .distance(90)
          .strength(0.4),
      )
      .force("charge", d3.forceManyBody().strength(-260))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide<SimNode>().radius(22));

    simulation.on("tick", () => {
      link
        .attr("x1", (d) => (d.source as SimNode).x ?? 0)
        .attr("y1", (d) => (d.source as SimNode).y ?? 0)
        .attr("x2", (d) => (d.target as SimNode).x ?? 0)
        .attr("y2", (d) => (d.target as SimNode).y ?? 0);
      node.attr("transform", (d) => `translate(${d.x ?? 0}, ${d.y ?? 0})`);
    });

    const drag = d3
      .drag<SVGGElement, SimNode>()
      .on("start", (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });

    node.call(drag as unknown as (sel: typeof node) => void);

    return () => {
      simulation.stop();
    };
  }, [nodes, links, centerId, height]);

  const typesPresent = Array.from(
    new Set(nodes.map((n) => n.entity_type).filter(Boolean)),
  );

  return (
    <div ref={containerRef} className="relative w-full">
      <svg
        ref={svgRef}
        role="img"
        aria-label="Knowledge graph subgraph"
        className="h-[520px] w-full rounded-lg border border-outline-variant/15 bg-surface-container-lowest/60"
      />
      {typesPresent.length > 0 ? (
        <div className="mt-2 flex flex-wrap gap-2 text-xs text-on-surface-variant">
          {typesPresent.map((t) => (
            <span key={t} className="inline-flex items-center gap-1">
              <span
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: colorFor(t) }}
              />
              {t}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}
