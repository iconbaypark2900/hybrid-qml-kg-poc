import { RankedForm } from "@/components/ranked-form";

export default function NewHypothesisPage() {
  return (
    <div className="space-y-6">
      <header>
        <h1 className="font-headline text-2xl font-semibold tracking-tight text-on-surface">
          New hypothesis
        </h1>
        <p className="mt-1 text-sm text-on-surface-variant">
          Rank intervention candidates using mechanism-informed hypotheses
          (H-001, H-002, H-003).
        </p>
      </header>
      <RankedForm />
    </div>
  );
}
