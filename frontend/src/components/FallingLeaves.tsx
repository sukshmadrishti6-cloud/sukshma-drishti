import React from 'react';

const leaves = Array.from({ length: 22 }, (_, index) => ({
  left: `${(index * 47) % 101}%`,
  delay: `${-((index * 1.73) % 14)}s`,
  duration: `${10 + ((index * 1.37) % 9)}s`,
  size: `${12 + ((index * 7) % 13)}px`,
  drift: `${-70 + ((index * 31) % 141)}px`,
  rotate: `${-45 + ((index * 29) % 91)}deg`,
  opacity: 0.18 + ((index * 13) % 22) / 100,
}));

export const FallingLeaves: React.FC = () => (
  <div className="sd-leaves" aria-hidden="true">
    {leaves.map((leaf, index) => (
      <span
        key={index}
        className="sd-leaf"
        style={{
          left: leaf.left,
          animationDelay: leaf.delay,
          animationDuration: leaf.duration,
          ['--leaf-size' as string]: leaf.size,
          ['--leaf-drift' as string]: leaf.drift,
          ['--leaf-rotate' as string]: leaf.rotate,
          opacity: leaf.opacity,
        }}
      >
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path
            d="M20.6 3.4C12.4 3.1 5.4 5.1 3.4 11.2c-1.1 3.4.6 6.6 3.8 7.3 3.2.7 6.1-1.1 7.3-4.3.9-2.3.8-5.4.7-6.8 1.9-.9 3.6-2.1 5.4-4Z"
            fill="currentColor"
          />
          <path d="M4.2 20.1c3.3-4.8 6.8-8.1 11.4-11" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
        </svg>
      </span>
    ))}
  </div>
);
