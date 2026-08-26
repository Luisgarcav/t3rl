import { ChartLineIcon, Code2Icon, Table2Icon } from "lucide-react";

import { Button } from "../../ui/button";

export type VisualizationMode = "visual" | "data" | "source";

const VISUALIZATION_MODES: ReadonlyArray<{
  readonly id: VisualizationMode;
  readonly label: string;
  readonly icon: typeof ChartLineIcon;
}> = [
  { id: "visual", label: "Visual", icon: ChartLineIcon },
  { id: "data", label: "Data", icon: Table2Icon },
  { id: "source", label: "Source", icon: Code2Icon },
];

export function VisualizationModeTabs({
  label,
  value,
  onChange,
}: {
  readonly label: string;
  readonly value: VisualizationMode;
  readonly onChange: (mode: VisualizationMode) => void;
}) {
  return (
    <div aria-label={label} className="flex w-fit gap-1 rounded-lg bg-muted/40 p-1" role="tablist">
      {VISUALIZATION_MODES.map((mode) => {
        const Icon = mode.icon;
        return (
          <Button
            aria-selected={value === mode.id}
            key={mode.id}
            role="tab"
            size="xs"
            variant={value === mode.id ? "secondary" : "ghost-muted"}
            onClick={() => onChange(mode.id)}
          >
            <Icon />
            {mode.label}
          </Button>
        );
      })}
    </div>
  );
}
