import { redirect } from "next/navigation";

/**
 * /molecular-design → /predict (permanent canonical redirect).
 * Both routes reach the same drug-disease pairwise prediction feature;
 * the canonical path is /predict.
 */
export default function MolecularDesignPage() {
  redirect("/predict");
}
