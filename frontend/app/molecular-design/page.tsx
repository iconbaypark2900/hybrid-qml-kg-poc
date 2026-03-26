import { PredictForm } from "@/components/predict-form";

export default function MolecularDesignPage() {
  return (
    <div className="space-y-6">
      <header>
        <h1 className="font-headline text-2xl font-semibold tracking-tight text-on-surface">
          Molecular design
        </h1>
        <p className="mt-1 text-sm text-on-surface-variant">
          Predict drug–disease treatment probability using the hybrid
          quantum-classical model.
        </p>
      </header>
      <PredictForm />
    </div>
  );
}
