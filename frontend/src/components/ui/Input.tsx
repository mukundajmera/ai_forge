import { InputHTMLAttributes, forwardRef, ReactNode } from 'react'
import clsx from 'clsx'

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
    label?: string;
    error?: string;
    hint?: string;
    icon?: ReactNode;
    rightIcon?: ReactNode;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
    ({ className, label, error, hint, icon, rightIcon, id, ...props }, ref) => {
        const inputId = id || `input-${Math.random().toString(36).substr(2, 9)}`;

        return (
            <div className="input-wrapper">
                {label && (
                    <label htmlFor={inputId} className="label">
                        {label}
                    </label>
                )}
                <div className="input-container" style={{ position: 'relative' }}>
                    {icon && (
                        <span className="input-icon input-icon-left">
                            {icon}
                        </span>
                    )}
                    <input
                        ref={ref}
                        id={inputId}
                        className={clsx(
                            'input',
                            error && 'input-error',
                            icon && 'input-with-left-icon',
                            rightIcon && 'input-with-right-icon',
                            className
                        )}
                        {...props}
                    />
                    {rightIcon && (
                        <span className="input-icon input-icon-right">
                            {rightIcon}
                        </span>
                    )}
                </div>
                {error && (
                    <span className="input-error-message" style={{
                        color: 'var(--error-500)',
                        fontSize: 'var(--text-xs)',
                        marginTop: 'var(--space-1)'
                    }}>
                        {error}
                    </span>
                )}
                {hint && !error && (
                    <span className="input-hint" style={{
                        color: 'var(--text-muted)',
                        fontSize: 'var(--text-xs)',
                        marginTop: 'var(--space-1)'
                    }}>
                        {hint}
                    </span>
                )}

                <style>{`
          .input-wrapper {
            width: 100%;
          }
          
          .input-container {
            position: relative;
            display: flex;
            align-items: center;
          }
          
          .input-icon {
            position: absolute;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
            pointer-events: none;
          }
          
          .input-icon-left {
            left: var(--space-3);
          }
          
          .input-icon-right {
            right: var(--space-3);
          }
          
          .input-with-left-icon {
            padding-left: 40px;
          }
          
          .input-with-right-icon {
            padding-right: 40px;
          }
        `}</style>
            </div>
        )
    }
)

Input.displayName = 'Input'

// Textarea component
interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
    label?: string;
    error?: string;
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
    ({ className, label, error, id, ...props }, ref) => {
        const textareaId = id || `textarea-${Math.random().toString(36).substr(2, 9)}`;

        return (
            <div className="input-wrapper">
                {label && (
                    <label htmlFor={textareaId} className="label">
                        {label}
                    </label>
                )}
                <textarea
                    ref={ref}
                    id={textareaId}
                    className={clsx('input', error && 'input-error', className)}
                    style={{ minHeight: 100, resize: 'vertical' }}
                    {...props}
                />
                {error && (
                    <span style={{
                        color: 'var(--error-500)',
                        fontSize: 'var(--text-xs)',
                        marginTop: 'var(--space-1)',
                        display: 'block'
                    }}>
                        {error}
                    </span>
                )}
            </div>
        )
    }
)

Textarea.displayName = 'Textarea'
