import { useEffect, useState } from 'react'

const STORAGE_KEY = 'theme'

export default function Header() {
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    return (localStorage.getItem(STORAGE_KEY) as 'light' | 'dark') || 'light'
  })

  useEffect(() => {
    document.body.classList.toggle('dark', theme === 'dark')
    localStorage.setItem(STORAGE_KEY, theme)
  }, [theme])

  const toggleTheme = () => {
    setTheme((t) => (t === 'light' ? 'dark' : 'light'))
  }

  return (
    <header className="p-4 border-b flex justify-end">
      <button onClick={toggleTheme}>Toggle Theme</button>
    </header>
  )
}
