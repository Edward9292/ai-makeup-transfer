/* Pipeline diagram: two lanes (extract, apply), drawn as an SVG batch-record flow.
   Pure presentational SVG — no client JS. */

const INK = "#17121A";
const PIGMENT = "#B0173F";
const ROUGE = "#C86A85";
const FAINT = "#DED2CF";

type NodeSpec = { x: number; label: string[]; accent?: boolean; checker?: boolean };

function Lane({
  y,
  title,
  nodes,
}: {
  y: number;
  title: string;
  nodes: NodeSpec[];
}) {
  const boxW = 134;
  const boxH = 54;
  return (
    <g>
      <text x={0} y={y - 14} className="pl-lane">
        {title}
      </text>
      {nodes.map((n, i) => (
        <g key={n.label.join()}>
          {i > 0 && (
            <line
              x1={nodes[i - 1].x + boxW}
              y1={y + boxH / 2}
              x2={n.x}
              y2={y + boxH / 2}
              stroke={FAINT}
              strokeWidth={1.5}
              markerEnd="url(#arrow)"
            />
          )}
          {n.checker && (
            <g>
              {[0, 1, 2, 3].map((r) =>
                [0, 1, 2, 3].map((c) =>
                  (r + c) % 2 === 0 ? (
                    <rect
                      key={`${r}-${c}`}
                      x={n.x + 4 + c * 6}
                      y={y + 4 + r * 6}
                      width={6}
                      height={6}
                      fill="#EFE6E3"
                    />
                  ) : null
                )
              )}
            </g>
          )}
          <rect
            x={n.x}
            y={y}
            width={boxW}
            height={boxH}
            rx={3}
            fill="none"
            stroke={n.accent ? PIGMENT : INK}
            strokeWidth={n.accent ? 2 : 1.2}
          />
          {n.label.map((line, li) => (
            <text
              key={line}
              x={n.x + boxW / 2}
              y={y + boxH / 2 + (li - (n.label.length - 1) / 2) * 14 + 4}
              textAnchor="middle"
              className={n.accent ? "pl-node pl-node-accent" : "pl-node"}
            >
              {line}
            </text>
          ))}
        </g>
      ))}
    </g>
  );
}

export default function Pipeline() {
  const xs = [70, 246, 422, 598, 774];
  return (
    <div className="pipeline-scroll">
      <svg
        viewBox="0 0 980 300"
        role="img"
        aria-label="Pipeline: a reference face is saved as a look; the target face is cropped, parsed, run through the transfer network, and the makeup difference is composited back at full resolution"
        className="pipeline-svg"
      >
        <defs>
          <marker id="arrow" viewBox="0 0 8 8" refX={7} refY={4} markerWidth={7} markerHeight={7} orient="auto">
            <path d="M0,0.5 L7,4 L0,7.5" fill="none" stroke={ROUGE} strokeWidth={1.4} />
          </marker>
        </defs>

        <Lane
          y={46}
          title="Create"
          nodes={[
            { x: xs[0], label: ["one photo", "anyone, with makeup"] },
            { x: xs[1], label: ["468 landmarks", "MediaPipe"] },
            { x: xs[2], label: ["face crop", "network framing"] },
            { x: xs[3], label: ["palette", "lips · lids · skin"] },
            { x: xs[4], label: ["saved look", "reference.png"], accent: true },
          ]}
        />

        {/* the layer is the artifact both lanes share */}
        <path
          d={`M ${xs[4] + 67} 100 L ${xs[4] + 67} 148 L ${xs[0] + 67} 148 L ${xs[0] + 67} 196`}
          fill="none"
          stroke={PIGMENT}
          strokeWidth={1.5}
          strokeDasharray="4 5"
          markerEnd="url(#arrow)"
        />
        <text x={(xs[0] + xs[4]) / 2 + 67} y={140} textAnchor="middle" className="pl-carry">
          one file — save it, share it, re-apply it
        </text>

        <Lane
          y={200}
          title="Try on"
          nodes={[
            { x: xs[0], label: ["look + your", "photo"], accent: true },
            { x: xs[1], label: ["crop + parse", "BiSeNet regions"] },
            { x: xs[2], label: ["CSD-MT", "makeup transfer"] },
            { x: xs[3], label: ["delta composite", "full-res, skin-gated"] },
            { x: xs[4], label: ["result", "result.png"] },
          ]}
        />
      </svg>
    </div>
  );
}
