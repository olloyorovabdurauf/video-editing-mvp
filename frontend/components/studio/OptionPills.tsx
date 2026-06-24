"use client";

export interface PillOption<T extends string> {
  value: T;
  label: string;
  hint?: string;
}

/**
 * Reusable labelled pill selector. Touch-friendly (large tap targets, wraps on
 * mobile) and keyboard-accessible. Used for language / content type / industry /
 * duration controls.
 */
export function OptionPills<T extends string>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: PillOption<T>[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div>
      <span className="label">{label}</span>
      <div className="flex flex-wrap gap-2">
        {options.map((o) => {
          const active = o.value === value;
          return (
            <button
              key={o.value}
              type="button"
              onClick={() => onChange(o.value)}
              aria-pressed={active}
              className={`rounded-xl border px-3.5 py-2.5 text-sm font-medium transition active:scale-[0.97]
                ${
                  active
                    ? "border-accent/60 bg-accent/15 text-white shadow-[0_0_0_1px_rgba(124,92,255,0.4)]"
                    : "border-white/10 bg-white/[0.03] text-white/60 hover:border-white/20 hover:text-white"
                }`}
            >
              {o.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
