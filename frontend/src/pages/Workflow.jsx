import { useState } from "react";
import StepIndicator from "../components/StepIndicator";
import UploadStep from "../components/workflow/UploadStep";
import PreprocessStep from "../components/workflow/PreprocessStep";
import TrainStep from "../components/workflow/TrainStep";
import EvaluateStep from "../components/workflow/EvaluateStep";
import PredictStep from "../components/workflow/PredictStep";

export default function Workflow() {
  const [step, setStep] = useState(1);
  const [dataset, setDataset] = useState(null);
  const [target, setTarget] = useState("");
  const [preprocessConfig, setPreprocessConfig] = useState(null);
  const [trainResult, setTrainResult] = useState(null);

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div>
        <h1 className="text-2xl font-extrabold text-slate-900 dark:text-white">ML Workflow</h1>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          From raw data to live predictions in five guided steps.
        </p>
      </div>

      <StepIndicator current={step} />

      <div className="card p-5 md:p-6">
        {step === 1 && (
          <UploadStep
            onComplete={({ dataset: ds, target: tg }) => {
              setDataset(ds);
              setTarget(tg);
              setPreprocessConfig(null);
              setTrainResult(null);
              setStep(2);
            }}
          />
        )}

        {step === 2 && dataset && (
          <PreprocessStep
            datasetId={dataset.id}
            target={target}
            onComplete={(config) => {
              setPreprocessConfig(config);
              setStep(3);
            }}
          />
        )}

        {step === 3 && dataset && (
          <TrainStep
            datasetId={dataset.id}
            target={target}
            preprocess={preprocessConfig || { mode: "auto", target_column: target }}
            onComplete={(result) => {
              setTrainResult(result);
              setStep(4);
            }}
          />
        )}

        {step === 4 && trainResult && (
          <EvaluateStep
            modelId={trainResult.model_id}
            modelType={trainResult.model_type}
            onComplete={() => setStep(5)}
          />
        )}

        {step === 5 && trainResult && (
          <PredictStep
            modelId={trainResult.model_id}
            dataset={dataset}
            target={target}
            classNames={trainResult.class_names}
          />
        )}
      </div>

      {step > 1 && (
        <div className="flex justify-between">
          <button className="btn-secondary" onClick={() => setStep((s) => Math.max(1, s - 1))}>
            ← Back
          </button>
          {step === 5 && (
            <button
              className="btn-primary"
              onClick={() => {
                setStep(1);
                setDataset(null);
                setTarget("");
                setPreprocessConfig(null);
                setTrainResult(null);
              }}
            >
              Start a new experiment
            </button>
          )}
        </div>
      )}
    </div>
  );
}
