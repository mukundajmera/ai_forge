import { forwardRef } from 'react'
import { Loader2 } from 'lucide-react'

type ButtonIntent = 'primary' | 'secondary' | 'ghost' | 'destructive'
type ButtonSize = 'sm' | 'md' | 'lg'

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    intent?: ButtonIntent
    size?: ButtonSize
    loading?: boolean
    icon?: React.ReactNode
    iconPosition?: 'left' | 'right'
    fullWidth?: boolean
}

const iconSizes: Record<ButtonSize, number> = {
    sm: 14,
    md: 16,
    lg: 18,
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
    ({
        className = '',
        intent = 'primary',
        size = 'md',
        loading,
        disabled,
        icon,
        iconPosition = 'left',
        fullWidth,
        children,
        ...props
    }, ref) => {
        const intentClass = `btn-${intent}`
        const sizeClass = `btn-${size}`
        const classes = [
            'btn',
            intentClass,
            sizeClass,
            fullWidth ? 'w-full' : '',
            className,
        ].filter(Boolean).join(' ')

        return (
            <button
                ref={ref}
                disabled={disabled || loading}
                className={classes}
                {...props}
            >
                {loading && <Loader2 className="spinner" size={iconSizes[size]} />}
                {!loading && icon && iconPosition === 'left' && icon}
                {children && <span>{children}</span>}
                {!loading && icon && iconPosition === 'right' && icon}
            </button>
        )
    }
)

Button.displayName = 'Button'

// Convenience exports for common button types
export function PrimaryButton(props: Omit<ButtonProps, 'intent'>) {
    return <Button intent="primary" {...props} />
}

export function SecondaryButton(props: Omit<ButtonProps, 'intent'>) {
    return <Button intent="secondary" {...props} />
}

export function GhostButton(props: Omit<ButtonProps, 'intent'>) {
    return <Button intent="ghost" {...props} />
}

export function DestructiveButton(props: Omit<ButtonProps, 'intent'>) {
    return <Button intent="destructive" {...props} />
}

export function IconButton({
    icon,
    'aria-label': ariaLabel,
    ...props
}: Omit<ButtonProps, 'children'> & { 'aria-label': string }) {
    return (
        <Button
            {...props}
            icon={icon}
            aria-label={ariaLabel}
            className={`btn-icon ${props.className || ''}`}
        />
    )
}
