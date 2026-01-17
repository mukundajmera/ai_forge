import { mockValidationResults } from '@/utils/mock-data'
import { Badge } from '@/components/ui/Badge'
import { formatRelativeTime } from '@/utils/formatters'
import { CheckCircle, XCircle } from 'lucide-react'

export function ValidationResultsTable() {
    return (
        <div className="table-container">
            <table className="table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Run Date</th>
                        <th>CodeBLEU</th>
                        <th>HumanEval</th>
                        <th>Perplexity</th>
                        <th>Latency</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {mockValidationResults.map((result) => (
                        <tr key={result.id}>
                            <td>
                                <span style={{ fontWeight: 'var(--font-semibold)' }}>
                                    {result.modelName}
                                </span>
                            </td>
                            <td style={{ color: 'var(--text-secondary)' }}>
                                {formatRelativeTime(result.runAt)}
                            </td>
                            <td>
                                <MetricCell value={result.metrics.codebleu} threshold={0.8} />
                            </td>
                            <td>
                                <MetricCell value={result.metrics.humaneval} threshold={0.7} isPercent />
                            </td>
                            <td>
                                <MetricCell
                                    value={result.metrics.perplexity}
                                    threshold={14}
                                    lowerIsBetter
                                />
                            </td>
                            <td>
                                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-sm)' }}>
                                    {result.metrics.latency}ms
                                </span>
                            </td>
                            <td>
                                {result.passed ? (
                                    <Badge variant="success">
                                        <CheckCircle size={12} /> Passed
                                    </Badge>
                                ) : (
                                    <Badge variant="danger">
                                        <XCircle size={12} /> Failed
                                    </Badge>
                                )}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    )
}

function MetricCell({
    value,
    threshold,
    lowerIsBetter = false,
    isPercent = false,
}: {
    value: number
    threshold: number
    lowerIsBetter?: boolean
    isPercent?: boolean
}) {
    const isGood = lowerIsBetter ? value <= threshold : value >= threshold
    const displayValue = isPercent ? `${(value * 100).toFixed(1)}%` : value.toFixed(2)

    return (
        <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 'var(--text-sm)',
            fontWeight: 'var(--font-semibold)',
            color: isGood ? 'var(--status-success)' : 'var(--text-primary)',
        }}>
            {displayValue}
        </span>
    )
}
