# Design System Specification: The Analytical Edge

## 1. Overview & Creative North Star
**Creative North Star: "The Precision Terminal"**

This design system is engineered to transform raw sports data into actionable intelligence. We are moving away from the loud, chaotic aesthetics of traditional sportsbooks and toward the authoritative, high-density environment of a quantitative trading floor. 

To achieve "The Precision Terminal" look, we reject the standard "boxed-in" dashboard grid. Instead, we utilize **intentional asymmetry** and **tonal layering** to guide the eye. Our layout strategy relies on overlapping "Glass" panels and depth-based hierarchy to ensure the platform feels like a sophisticated tool, not a game. We prioritize data density without sacrificing clarity, using high-contrast typography scales and surgical precision in our spacing.

---

## 2. Colors & Surface Logic

Our palette is anchored in deep charcoal and navy to minimize eye strain during long-form analysis, with electric accents that signal technical power.

### Surface Hierarchy & The "No-Line" Rule
Traditional 1px borders are prohibited for sectioning. They create visual noise and "trap" data. 
- **Containment Strategy:** Define boundaries exclusively through background color shifts. Use `surface-container-low` for primary sections and `surface-container-highest` for nested data modules.
- **Nesting:** Treat the UI as physical layers. A `surface-container-lowest` card should sit atop a `surface-container-low` section to create a natural, soft lift.
- **The Glass & Gradient Rule:** For floating modals or "Value Score" breakdowns, use Glassmorphism: semi-transparent `surface-variant` colors with a `20px` backdrop-blur. 
- **Signature CTA:** Use a linear gradient from `primary` (#a4e6ff) to `primary-container` (#00d1ff) at a 135° angle to give action elements a "pulsing" digital soul.

| Token | Hex | Role |
| :--- | :--- | :--- |
| `background` | #10131a | Main canvas |
| `primary` | #a4e6ff | "The Line" accents, active states |
| `tertiary` | #66f796 | "Money Green" Value Indicator (70+) |
| `surface-container` | #1d2026 | Secondary modules |
| `error` | #ffb4ab | Critical alerts only (Avoid for losses) |

---

## 3. Typography

We use a dual-font strategy to balance editorial authority with technical precision.

- **Display & Headlines (Space Grotesk):** This typeface provides a geometric, modern "tech" feel. Use `display-lg` for high-level market boards to establish a bold, editorial presence.
- **Data & Body (Inter):** Chosen for its exceptional legibility at small sizes. All numerical data points and "Value Scores" must use Inter with a slightly tighter letter-spacing (-0.02em) to mimic terminal readouts.
- **The "Data Hero" Pattern:** When displaying "The Line" or a "Value Score," use `headline-lg` in `tertiary` (#66f796) to ensure the most critical intelligence is the first thing a user sees.

---

## 4. Elevation & Depth

We eschew traditional shadows in favor of **Tonal Layering**.

- **The Layering Principle:** Depth is achieved by stacking. A card in `surface-container-lowest` on a `surface` background creates a "sunken" technical feel. A card in `surface-container-highest` creates an "elevated" prominent feel.
- **Ambient Shadows:** For floating elements (e.g., tooltips, filter dropdowns), use a wide-spread shadow: `0px 12px 32px rgba(0, 0, 0, 0.4)`. The shadow must be a tinted version of the background to feel integrated.
- **The "Ghost Border" Fallback:** If accessibility requires a border, use `outline-variant` at **15% opacity**. This provides a "suggestion" of a boundary without breaking the fluid layout.
- **Motion-Depth:** Elements should appear to slide "under" the glass header during scroll, utilizing the backdrop-blur effect of the header container.

---

## 5. Components

### The "Value Score" Badge
*   **Design:** A pill-shaped container (`rounded-full`) using `tertiary-container` for scores >70. 
*   **Typography:** `label-md` bold. 
*   **Interaction:** On hover, trigger a Glassmorphic tooltip explaining the AI logic behind the score.

### Buttons
*   **Primary:** Gradient fill (`primary` to `primary-container`), `rounded-md`, white text. No border.
*   **Secondary:** Ghost style. Transparent fill, `primary` 20% opacity "Ghost Border," `primary` text.
*   **Tertiary:** Text-only with an underline that appears only on hover, mimicking a terminal command.

### Data Inputs & Search
*   **Style:** `surface-container-highest` background, `rounded-sm`. 
*   **State:** On focus, the background transitions to `surface-bright` with a 1px `primary` bottom-border (The "Line" indicator).

### Lists & Cards
*   **Constraint:** **Zero dividers.** Separate list items using `12px` of vertical white space or alternating backgrounds between `surface-container-low` and `surface-container-lowest`. 
*   **Visual Soul:** Incorporate a subtle "edge" glow on the left side of cards using a 2px `primary` strip to indicate high-value opportunities.

---

## 6. Do’s and Don'ts

### Do
*   **Do** use `Roboto Mono` or `Inter` for all "Value Score" numbers to emphasize technical accuracy.
*   **Do** use asymmetrical layouts where the main "Edge" data is prioritized over the secondary "Market" data.
*   **Do** utilize `surface-container` tiers to create a logical "information scent" from most important to least important.

### Don't
*   **Don't** use casino-style red/gold. We are an intelligence platform, not a gambling hall. Use `error` (#ffb4ab) only for system failures, not lost bets.
*   **Don't** use 100% opaque, high-contrast borders. This shatters the "Precision Terminal" aesthetic.
*   **Don't** use standard "Drop Shadows" (e.g., 2px blur, 50% opacity). They feel "templated" and cheap. Use ambient, wide-spread blurs only.
*   **Don't** crowd the interface. Data density is good, but "The Line" needs breathing room to be found. Use `xl` (0.75rem) roundedness sparingly on major containers to soften the technical edge.