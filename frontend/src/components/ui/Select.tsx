import { SelectHTMLAttributes, forwardRef } from 'react'

interface SelectOption {
    value: string
    label: string
    disabled?: boolean
}

interface SelectProps extends Omit<SelectHTMLAttributes<HTMLSelectElement>, 'size'> {
    options: SelectOption[]
    size?: 'sm' | 'md' | 'lg'
    error?: boolean
    placeholder?: string
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
    ({ options, size = 'md', error, placeholder, className = '', ...props }, ref) => {
        const sizeStyles = {
            sm: { height: 32, fontSize: 13 },
            md: { height: 40, fontSize: 'var(--text-sm)' },
            lg: { height: 48, fontSize: 'var(--text-base)' },
        }

        return (
            <select
                ref={ref}
                className={`input ${error ? 'input-error' : ''} ${className}`}
                style={{
                    ...sizeStyles[size],
                    cursor: 'pointer',
                    appearance: 'none',
                    backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12' fill='none'%3E%3Cpath d='M2.5 4.5L6 8L9.5 4.5' stroke='%23a1a1aa' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E")`,
                    backgroundRepeat: 'no-repeat',
                    backgroundPosition: 'right 12px center',
                    paddingRight: 'var(--space-10)',
                }}
                {...props}
            >
                {placeholder && (
                    <option value="" disabled>
                        {placeholder}
                    </option>
                )}
                {options.map((option) => (
                    <option
                        key={option.value}
                        value={option.value}
                        disabled={option.disabled}
                    >
                        {option.label}
                    </option>
                ))}
            </select>
        )
    }
)

Select.displayName = 'Select'
