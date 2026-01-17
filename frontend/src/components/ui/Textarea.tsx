import { TextareaHTMLAttributes, forwardRef } from 'react'

interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
    error?: boolean
    resize?: 'none' | 'vertical' | 'horizontal' | 'both'
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
    ({ error, resize = 'vertical', className = '', style, ...props }, ref) => {
        return (
            <textarea
                ref={ref}
                className={`input ${error ? 'input-error' : ''} ${className}`}
                style={{
                    height: 'auto',
                    minHeight: 100,
                    padding: 'var(--space-3)',
                    resize,
                    lineHeight: 'var(--leading-relaxed)',
                    ...style,
                }}
                {...props}
            />
        )
    }
)

Textarea.displayName = 'Textarea'
