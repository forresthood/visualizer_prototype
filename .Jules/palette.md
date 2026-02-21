## 2025-05-23 - Accessibility First: Label Buddies and Mnemonics
**Learning:** PyQt6 labels (`QLabel`) do not automatically associate with their neighboring controls. Explicitly using `setBuddy()` and adding mnemonics (e.g., `&Mode`) allows for keyboard shortcuts (Alt+M) and screen reader support, significantly improving accessibility with minimal code changes.
**Action:** Always verify if form labels have associated buddies and keyboard shortcuts. Add tooltips to controls to further aid understanding and discoverability of shortcuts.
