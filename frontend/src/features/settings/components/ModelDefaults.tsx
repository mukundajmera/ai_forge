import { useState } from 'react'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Select } from '@/components/ui/Select'
import { Save, RotateCcw } from 'lucide-react'

export function ModelDefaults() {
    const [epochs, setEpochs] = useState('3')
    const [learningRate, setLearningRate] = useState('0.0002')
    const [batchSize, setBatchSize] = useState('2')
    const [rank, setRank] = useState('64')
    const [method, setMethod] = useState('pissa')
    const [baseModel, setBaseModel] = useState('llama-3.2-3b')
    const [quantization, setQuantization] = useState('q4_k_m')

    return (
        <div className="settings-section">
            <div className="section-header">
                <h2>Model Training Defaults</h2>
                <p>Set default configuration for new training jobs</p>
            </div>

            <div className="settings-grid">
                {/* Base Model */}
                <div className="setting-row">
                    <label className="setting-label">Base Model</label>
                    <Select
                        value={baseModel}
                        onChange={(e) => setBaseModel(e.target.value)}
                        options={[
                            { value: 'llama-3.2-3b', label: 'Llama 3.2 3B' },
                            { value: 'llama-3.2-7b', label: 'Llama 3.2 7B' },
                            { value: 'codellama-7b', label: 'CodeLlama 7B' },
                            { value: 'mistral-7b', label: 'Mistral 7B' },
                        ]}
                    />
                </div>

                {/* Training Method */}
                <div className="setting-row">
                    <label className="setting-label">Training Method</label>
                    <Select
                        value={method}
                        onChange={(e) => setMethod(e.target.value)}
                        options={[
                            { value: 'pissa', label: 'PiSSA' },
                            { value: 'lora', label: 'LoRA' },
                            { value: 'qlora', label: 'QLoRA' },
                        ]}
                    />
                </div>

                <div className="divider" />

                {/* Hyperparameters */}
                <div className="hyperparam-grid">
                    <div className="setting-row compact">
                        <label className="setting-label">Epochs</label>
                        <Input
                            type="number"
                            value={epochs}
                            onChange={(e) => setEpochs(e.target.value)}
                            min={1}
                            max={10}
                        />
                    </div>

                    <div className="setting-row compact">
                        <label className="setting-label">Learning Rate</label>
                        <Input
                            type="text"
                            value={learningRate}
                            onChange={(e) => setLearningRate(e.target.value)}
                        />
                    </div>

                    <div className="setting-row compact">
                        <label className="setting-label">Batch Size</label>
                        <Input
                            type="number"
                            value={batchSize}
                            onChange={(e) => setBatchSize(e.target.value)}
                            min={1}
                            max={8}
                        />
                    </div>

                    <div className="setting-row compact">
                        <label className="setting-label">Rank</label>
                        <Input
                            type="number"
                            value={rank}
                            onChange={(e) => setRank(e.target.value)}
                            min={8}
                            max={256}
                        />
                    </div>
                </div>

                <div className="divider" />

                {/* Output Settings */}
                <div className="setting-row">
                    <label className="setting-label">Quantization</label>
                    <Select
                        value={quantization}
                        onChange={(e) => setQuantization(e.target.value)}
                        options={[
                            { value: 'q4_k_m', label: 'Q4_K_M (Recommended)' },
                            { value: 'q5_k_m', label: 'Q5_K_M (Higher Quality)' },
                            { value: 'q8_0', label: 'Q8_0 (Best Quality)' },
                            { value: 'f16', label: 'F16 (Full Precision)' },
                        ]}
                    />
                </div>
            </div>

            <div className="section-footer">
                <Button intent="secondary" icon={<RotateCcw size={16} />}>
                    Reset to Defaults
                </Button>
                <Button icon={<Save size={16} />}>
                    Save Defaults
                </Button>
            </div>

            <style>{`
                .settings-section {
                    background: var(--bg-surface);
                    border: 1px solid var(--border-subtle);
                    border-radius: var(--radius-lg);
                    padding: var(--space-6);
                }

                .section-header {
                    margin-bottom: var(--space-6);
                    padding-bottom: var(--space-4);
                    border-bottom: 1px solid var(--border-subtle);
                }

                .section-header h2 {
                    font-size: var(--text-lg);
                    font-weight: var(--font-semibold);
                    margin: 0 0 var(--space-1) 0;
                }

                .section-header p {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                    margin: 0;
                }

                .settings-grid {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-4);
                }

                .setting-row {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-2);
                }

                .setting-row.compact {
                    gap: var(--space-1);
                }

                .setting-label {
                    font-size: var(--text-sm);
                    font-weight: var(--font-medium);
                    color: var(--text-primary);
                }

                .divider {
                    height: 1px;
                    background: var(--border-subtle);
                    margin: var(--space-2) 0;
                }

                .hyperparam-grid {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: var(--space-4);
                }

                .section-footer {
                    display: flex;
                    justify-content: flex-end;
                    gap: var(--space-2);
                    margin-top: var(--space-6);
                    padding-top: var(--space-4);
                    border-top: 1px solid var(--border-subtle);
                }

                @media (max-width: 640px) {
                    .hyperparam-grid {
                        grid-template-columns: 1fr;
                    }
                }
            `}</style>
        </div>
    )
}
