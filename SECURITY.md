# Security and data handling

The benchmark is a local research tool. It downloads the known UCI archive over HTTPS, verifies a pinned SHA-256 and size, and accepts only the expected regular files during extraction. Treat externally supplied sensor logs as data and keep them separate from the verified archive directory.

Report security issues through this repository's private vulnerability reporting feature if available. If it is unavailable, open an issue requesting a private contact channel and omit any sensitive files, personal information, or exploit details.

Keep raw sensor logs and archives out of Git and release bundles. Use the synthetic CLI or a generated fixture when reporting parsing or evaluation problems. The benchmark provides no clinical assessment or identity inference.
