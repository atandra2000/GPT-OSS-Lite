/* GPT-OSS-Lite Documentation Portal Client-Side Behaviors
   ======================================================================
   • Attention-Sink Gravity Well & YaRN 128K Resonance (FIG · A0)
   • Live 12-Layer Pipeline Pass Telemetry (FIG · A1)
   • 12-Layer Stack Architecture Explorer (FIG · 01)
   • MoE Grouped-GEMM Routing Playground (FIG · C4)
   • Per-Head Attention-Sink Bias Clamp Explorer (FIG · C2)
   • Expandable Code Blocks (>14 lines) & Copy-to-Clipboard
   • Navigation Filtering & Sidebar Mobile Toggle
   • Table of Contents Scrollspy
   • Highlight.js & KaTeX bootstrap
*/

(function () {
    'use strict';

    var reduced = window.matchMedia &&
                  window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // ------------------------------------------------------------------
    // 1. Attention-Sink Gravity Well & YaRN 128K Resonance (FIG · A0)
    //    Interactive high-DPI canvas simulation of GPT-OSS attention:
    //    - Central attention sink gravity well (virtual per-head logit, clamped)
    //    - Sliding-window (W=128) ring buffer orbital sector
    //    - 36 YaRN multi-frequency RoPE rotary phasors (θ=100K, scale=32)
    //    - Pruned RoPE 25% active subspace on global layers
    //    - Inflowing geodesic spiral quanta and dynamic attention chords
    //    - Vector flow field of attention gradients and sink potentials
    //    - Interactive query pulse shockwave on click & HUD hover reticle
    // ------------------------------------------------------------------
    function initAttentionSinkHero() {
        var canvas = document.getElementById('attentionSinkCanvas');
        if (!canvas) return;

        var ctx = canvas.getContext('2d');
        if (!ctx) return;

        var container = canvas.parentElement;
        var probeEl = document.getElementById('hudProbe');
        var probeTag = document.getElementById('probeTag');
        var probeCoords = document.getElementById('probeCoords');
        var probeDecay = document.getElementById('probeDecay');
        var normValEl = document.getElementById('hudNormVal');
        var pauseBtn = document.getElementById('figPauseBtn');
        var modeBtns = document.querySelectorAll('.hero-figure-decode .fig-controls .fig-btn[data-mode]');

        var currentMode = 'sink';
        var isPaused = false;
        var isHovered = false;
        var mouseX = -9999, mouseY = -9999;
        var probeNode = null;
        var pulseWaves = [];
        var animId = null;
        var lastTime = 0;
        var simTime = 0;

        // Palette tokens matching the dark bench notebook theme
        var PALETTE = {
            paperBg: '#0e0c0a',
            paperCenter: '#17130f',
            gridRule: 'rgba(58, 50, 38, 0.45)',
            gridAxis: 'rgba(201, 163, 92, 0.35)',
            unitCircle: 'rgba(201, 163, 92, 0.22)',
            olive: '#9a9440',
            oliveGlow: 'rgba(154, 148, 64, 0.65)',
            oliveTint: 'rgba(154, 148, 64, 0.15)',
            terracotta: '#e07a3f',
            terracottaGlow: 'rgba(224, 122, 63, 0.65)',
            terracottaTint: 'rgba(224, 122, 63, 0.15)',
            gold: '#c9a35c',
            goldGlow: 'rgba(201, 163, 92, 0.75)',
            goldTint: 'rgba(201, 163, 92, 0.18)',
            coreHot: '#fffaf0',
            ink: '#d8ccb4',
            inkFaint: '#7a7160'
        };

        // 4 Initial Sink Tokens (k=0..3) with learned sink bias parameters
        var sinkTokens = [
            { id: 0, bias: 4.18, weight: 0.342, rFrac: 0.05, angle: 0, label: 'SINK #0 (Anchor)' },
            { id: 1, bias: 2.85, weight: 0.186, rFrac: 0.09, angle: Math.PI * 0.5, label: 'SINK #1' },
            { id: 2, bias: 1.94, weight: 0.112, rFrac: 0.13, angle: Math.PI * 1.0, label: 'SINK #2' },
            { id: 3, bias: 1.25, weight: 0.078, rFrac: 0.17, angle: Math.PI * 1.5, label: 'SINK #3' }
        ];

        // 36 Sequence Token Phasors across the logarithmic manifold
        var PHASORS_COUNT = 36;
        var phasors = [];
        for (var i = 0; i < PHASORS_COUNT; i++) {
            var frac = i / (PHASORS_COUNT - 1);
            var baseRadius = 0.22 + 0.74 * Math.pow(frac, 0.85);
            var omega = 0.25 + 1.75 * Math.pow(1 - frac, 1.2);
            var phase0 = (i * 2.399963229728653) % (Math.PI * 2);
            var isPruned = (i % 4 !== 0);
            var isSink = (i < 4);

            phasors.push({
                id: i,
                tokenIdx: Math.round(Math.pow(frac, 2.4) * 128000),
                r: baseRadius,
                omega: omega,
                phase: phase0,
                theta: phase0,
                x: 0, y: 0,
                trail: [],
                glow: 0,
                isPruned: isPruned,
                isSink: isSink,
                yarnScale: frac < 0.25 ? 1.0 : (1.0 + 31.0 * Math.pow((frac - 0.25) / 0.75, 1.5))
            });
        }

        // 32 Logarithmic Spiral Quanta (Inflowing Attention Energy Stream)
        var PARTICLES_COUNT = 32;
        var particles = [];
        for (var p = 0; p < PARTICLES_COUNT; p++) {
            particles.push({
                arm: p % 4,
                t: (p / PARTICLES_COUNT) * 4.0,
                speed: 0.22 + 0.18 * Math.random(),
                size: 1.2 + 1.2 * Math.random(),
                alpha: 0.3 + 0.7 * Math.random()
            });
        }

        // Viewport and Geometry Dimensions
        var width = 0, height = 0, cx = 0, cy = 0, maxR = 0;
        function resize() {
            var rect = container.getBoundingClientRect();
            var dpr = Math.min(window.devicePixelRatio || 1, 2);
            width = rect.width;
            height = rect.height || 360;
            canvas.width = Math.round(width * dpr);
            canvas.height = Math.round(height * dpr);
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.scale(dpr, dpr);
            cx = width / 2;
            cy = height / 2;
            maxR = Math.min(width * 0.38, height * 0.38);
        }

        if (window.ResizeObserver) {
            var ro = new ResizeObserver(function () {
                resize();
                if (reduced || isPaused) renderFrame(0, true);
            });
            ro.observe(container);
        } else {
            window.addEventListener('resize', resize);
        }
        resize();

        // Trigger Query Excitation Shockwave
        function triggerPulse(originX, originY) {
            var ox = originX !== undefined ? originX : cx;
            var oy = originY !== undefined ? originY : cy;
            pulseWaves.push({
                x: ox,
                y: oy,
                r: 0,
                maxR: Math.max(width, height) * 0.85,
                speed: 380,
                alpha: 1.0
            });
            phasors.forEach(function (ph) {
                ph.glow = 1.0;
            });
        }

        // Event Listeners
        container.addEventListener('mousemove', function (e) {
            var rect = canvas.getBoundingClientRect();
            mouseX = e.clientX - rect.left;
            mouseY = e.clientY - rect.top;
            isHovered = true;
        });

        container.addEventListener('mouseleave', function () {
            isHovered = false;
            mouseX = -9999;
            mouseY = -9999;
            probeNode = null;
            if (probeEl) probeEl.style.opacity = '0';
        });

        container.addEventListener('click', function (e) {
            var rect = canvas.getBoundingClientRect();
            var clickX = e.clientX - rect.left;
            var clickY = e.clientY - rect.top;
            triggerPulse(clickX, clickY);
        });

        if (pauseBtn) {
            pauseBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                isPaused = !isPaused;
                pauseBtn.innerHTML = isPaused ? '&#9658;' : '&#10074;&#10074;';
                pauseBtn.setAttribute('aria-label', isPaused ? 'Resume animation' : 'Pause animation');
                if (!isPaused && !animId) {
                    lastTime = performance.now();
                    loop(lastTime);
                }
            });
        }

        modeBtns.forEach(function (btn) {
            btn.addEventListener('click', function (e) {
                e.stopPropagation();
                modeBtns.forEach(function (b) { b.classList.remove('active'); });
                btn.classList.add('active');
                currentMode = btn.getAttribute('data-mode') || 'sink';
                if (reduced || isPaused) renderFrame(0, true);
            });
        });

        // --------------------------------------------------------------
        // Render Functions
        // --------------------------------------------------------------

        function drawBackground() {
            var grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, maxR * 1.35);
            grad.addColorStop(0, PALETTE.paperCenter);
            grad.addColorStop(0.65, PALETTE.paperBg);
            grad.addColorStop(1, '#070605');
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, width, height);

            ctx.save();
            ctx.lineWidth = 1;

            // Outer dial perimeter tick marks (64 precision ticks)
            var totalTicks = 64;
            for (var tk = 0; tk < totalTicks; tk++) {
                var tkAng = (tk * Math.PI * 2) / totalTicks;
                var isMajor = tk % 8 === 0;
                var isSemi = tk % 4 === 0;
                var rInner = maxR * (isMajor ? 0.96 : (isSemi ? 0.975 : 0.985));
                var rOuter = maxR * 1.0;

                ctx.beginPath();
                ctx.moveTo(cx + rInner * Math.cos(tkAng), cy + rInner * Math.sin(tkAng));
                ctx.lineTo(cx + rOuter * Math.cos(tkAng), cy + rOuter * Math.sin(tkAng));
                ctx.strokeStyle = isMajor ? PALETTE.gold : (isSemi ? 'rgba(201, 163, 92, 0.45)' : 'rgba(58, 50, 38, 0.4)');
                ctx.lineWidth = isMajor ? 1.2 : 0.8;
                ctx.stroke();
            }

            // Concentric Horizon Rings
            var rings = [
                { frac: 0.18, label: 'SINK Φ', dash: [2, 3], col: PALETTE.gold },
                { frac: 0.48, label: 'W=128 SWA', dash: [3, 4], col: PALETTE.olive },
                { frac: 0.76, label: 'YaRN RAMP', dash: [2, 5], col: PALETTE.terracotta },
                { frac: 1.00, label: '128K CTX', dash: [4, 4], col: PALETTE.unitCircle }
            ];
            var angLabel = -Math.PI * 0.22;

            rings.forEach(function (rg, idx) {
                var r = maxR * rg.frac;
                ctx.beginPath();
                ctx.arc(cx, cy, r, 0, Math.PI * 2);
                ctx.strokeStyle = rg.col;
                ctx.setLineDash(rg.dash);
                ctx.lineWidth = idx === 0 || idx === rings.length - 1 ? 1.1 : 0.8;
                ctx.stroke();

                ctx.setLineDash([]);
                var lx = cx + r * Math.cos(angLabel);
                var ly = cy + r * Math.sin(angLabel);
                ctx.fillStyle = rg.col;
                ctx.font = 'bold 9px "JetBrains Mono", monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(rg.label, lx + 6, ly - 4);
            });

            // Crosshair Axes (+Q, -Q, +K, -K)
            ctx.setLineDash([]);
            ctx.strokeStyle = PALETTE.gridAxis;

            ctx.beginPath();
            ctx.moveTo(cx - maxR * 1.04, cy);
            ctx.lineTo(cx + maxR * 1.04, cy);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(cx, cy - maxR * 1.04);
            ctx.lineTo(cx, cy + maxR * 1.04);
            ctx.stroke();

            ctx.fillStyle = PALETTE.gold;
            ctx.font = 'bold 10px "JetBrains Mono", monospace';
            ctx.textAlign = 'left';
            ctx.fillText('+Q', cx + maxR * 1.04 + 4, cy + 3);
            ctx.textAlign = 'right';
            ctx.fillText('-Q', cx - maxR * 1.04 - 4, cy + 3);

            ctx.textAlign = 'center';
            ctx.fillText('+K', cx, cy - maxR * 1.04 - 4);
            ctx.fillText('-K', cx, cy + maxR * 1.04 + 12);

            ctx.strokeStyle = 'rgba(58, 50, 38, 0.22)';
            ctx.setLineDash([1, 6]);
            [Math.PI / 6, Math.PI / 4, Math.PI / 3, Math.PI * 2 / 3, Math.PI * 3 / 4, Math.PI * 5 / 6].forEach(function (ang) {
                ctx.beginPath();
                ctx.moveTo(cx - Math.cos(ang) * maxR, cy - Math.sin(ang) * maxR);
                ctx.lineTo(cx + Math.cos(ang) * maxR, cy + Math.sin(ang) * maxR);
                ctx.stroke();
            });
            ctx.setLineDash([]);

            ctx.restore();
        }

        // Mode 1: Logarithmic Spiral Manifolds & Particles
        function drawSpiralManifolds(t) {
            ctx.save();
            var arms = 4;
            var rot = t * 0.12;

            for (var a = 0; a < arms; a++) {
                var armOffset = (a * Math.PI * 2) / arms + rot;
                ctx.beginPath();
                var started = false;

                for (var theta = 0; theta < Math.PI * 3.6; theta += 0.08) {
                    var rNorm = Math.exp(0.38 * (theta - Math.PI * 3.6));
                    var r = rNorm * maxR * 1.05;
                    var angle = armOffset - theta;
                    var x = cx + r * Math.cos(angle);
                    var y = cy + r * Math.sin(angle);

                    if (!started) {
                        ctx.moveTo(x, y);
                        started = true;
                    } else {
                        ctx.lineTo(x, y);
                    }
                }

                var grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, maxR);
                grad.addColorStop(0, 'rgba(201, 163, 92, 0.45)');
                grad.addColorStop(0.5, 'rgba(154, 148, 64, 0.25)');
                grad.addColorStop(1, 'rgba(44, 38, 28, 0.05)');
                ctx.strokeStyle = grad;
                ctx.lineWidth = 1.2;
                ctx.stroke();
            }

            particles.forEach(function (pt) {
                if (!reduced) pt.t += 0.015 * pt.speed;
                if (pt.t > 3.6) pt.t = 0.0;

                var theta = pt.t * Math.PI;
                var rNorm = Math.exp(0.38 * (theta - Math.PI * 3.6));
                var r = rNorm * maxR * 1.05;
                var angle = (pt.arm * Math.PI * 2) / arms + rot - theta;
                var px = cx + r * Math.cos(angle);
                var py = cy + r * Math.sin(angle);

                ctx.beginPath();
                ctx.arc(px, py, pt.size, 0, Math.PI * 2);
                ctx.fillStyle = pt.arm % 2 === 0 ? PALETTE.goldGlow : PALETTE.oliveGlow;
                ctx.fill();
            });

            ctx.restore();
        }

        // Mode 1: Sink Gravity Well & Sliding-Window Orbits
        function drawSinkGravityWell(t, dt) {
            ctx.save();

            // 1. Central Sink Corona Potential Well
            var sinkR = maxR * 0.18;
            var coronaGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, sinkR * 1.5);
            coronaGrad.addColorStop(0, 'rgba(255, 250, 240, 0.85)');
            coronaGrad.addColorStop(0.25, PALETTE.goldGlow);
            coronaGrad.addColorStop(0.65, PALETTE.goldTint);
            coronaGrad.addColorStop(1, 'rgba(201, 163, 92, 0)');
            ctx.fillStyle = coronaGrad;
            ctx.beginPath();
            ctx.arc(cx, cy, sinkR * 1.5, 0, Math.PI * 2);
            ctx.fill();

            // Pulsating Sink Core Equipotential Line
            var pulseOffset = Math.sin(t * 2.5) * 2.5;
            ctx.beginPath();
            ctx.arc(cx, cy, sinkR + pulseOffset, 0, Math.PI * 2);
            ctx.strokeStyle = PALETTE.gold;
            ctx.lineWidth = 1.2;
            ctx.setLineDash([2, 4]);
            ctx.stroke();
            ctx.setLineDash([]);

            // 2. Active Query Token & Sliding-Window Annular Wedge
            var qAngle = (t * 0.35) % (Math.PI * 2);
            var qRadius = maxR * 0.48; // on SWA track
            var qX = cx + qRadius * Math.cos(qAngle);
            var qY = cy - qRadius * Math.sin(qAngle);

            // Draw SWA Window Annular Sector (W=128 tokens window)
            var winSpan = 0.55; // angular span of sliding window
            ctx.beginPath();
            ctx.arc(cx, cy, maxR * 0.56, -qAngle - winSpan, -qAngle + 0.1, false);
            ctx.arc(cx, cy, maxR * 0.20, -qAngle + 0.1, -qAngle - winSpan, true);
            ctx.closePath();
            ctx.fillStyle = 'rgba(154, 148, 64, 0.12)';
            ctx.fill();
            ctx.strokeStyle = 'rgba(154, 148, 64, 0.45)';
            ctx.lineWidth = 0.9;
            ctx.stroke();

            // 3. Update Phasor Positions & Trails
            var closestNode = null;
            var closestDist = 99999;

            // Check distance to Sink Tokens
            sinkTokens.forEach(function (st) {
                var sRadius = maxR * st.rFrac;
                var sAng = st.angle + t * (0.4 / (st.id + 1));
                st.x = cx + sRadius * Math.cos(sAng);
                st.y = cy - sRadius * Math.sin(sAng);

                if (isHovered) {
                    var d = Math.hypot(mouseX - st.x, mouseY - st.y);
                    if (d < closestDist && d < 40) {
                        closestDist = d;
                        closestNode = { type: 'sink', data: st };
                    }
                }
            });

            // Update sequence phasors
            phasors.forEach(function (ph) {
                if (!reduced) {
                    ph.theta = ph.phase + ph.omega * t;
                    if (ph.glow > 0) ph.glow = Math.max(0, ph.glow - dt * 1.8);
                }

                var r = ph.r * maxR;
                ph.x = cx + r * Math.cos(ph.theta);
                ph.y = cy - r * Math.sin(ph.theta);

                if (!reduced) {
                    ph.trail.unshift({ x: ph.x, y: ph.y, a: 1.0 });
                    if (ph.trail.length > 18) ph.trail.pop();
                }

                if (isHovered) {
                    var d = Math.hypot(mouseX - ph.x, mouseY - ph.y);
                    if (d < closestDist && d < 40) {
                        closestDist = d;
                        closestNode = { type: 'phasor', data: ph };
                    }
                }
            });

            probeNode = closestNode;

            // 4. Draw Attention Sink Tether Laser Beams (Query + Window Tokens -> Sink 0)
            var s0 = sinkTokens[0];
            ctx.beginPath();
            ctx.moveTo(qX, qY);
            ctx.lineTo(s0.x, s0.y);
            ctx.strokeStyle = PALETTE.goldGlow;
            ctx.lineWidth = 1.6;
            ctx.stroke();

            // Tethers from window tokens to sink
            phasors.forEach(function (ph) {
                var dAngle = Math.abs((ph.theta % (Math.PI * 2)) - qAngle);
                if (dAngle < winSpan || Math.abs(dAngle - Math.PI * 2) < winSpan) {
                    ctx.beginPath();
                    ctx.moveTo(ph.x, ph.y);
                    ctx.lineTo(s0.x, s0.y);
                    ctx.strokeStyle = 'rgba(201, 163, 92, 0.22)';
                    ctx.lineWidth = 0.8;
                    ctx.setLineDash([2, 4]);
                    ctx.stroke();
                    ctx.setLineDash([]);

                    // Local attention connection to Query
                    ctx.beginPath();
                    ctx.moveTo(ph.x, ph.y);
                    ctx.lineTo(qX, qY);
                    ctx.strokeStyle = 'rgba(154, 148, 64, 0.32)';
                    ctx.lineWidth = 1.0;
                    ctx.stroke();
                }
            });

            // 5. Draw Fading Orbital Trails
            phasors.forEach(function (ph) {
                if (ph.trail.length > 1) {
                    for (var k = 0; k < ph.trail.length - 1; k++) {
                        var alpha = (1 - (k / ph.trail.length)) * 0.5;
                        ctx.beginPath();
                        ctx.moveTo(ph.trail[k].x, ph.trail[k].y);
                        ctx.lineTo(ph.trail[k + 1].x, ph.trail[k + 1].y);
                        ctx.strokeStyle = ph.isSink ? 'rgba(201, 163, 92, ' + alpha.toFixed(3) + ')' :
                                         (ph.r < 0.48 ? 'rgba(154, 148, 64, ' + alpha.toFixed(3) + ')' :
                                                        'rgba(224, 122, 63, ' + alpha.toFixed(3) + ')');
                        ctx.lineWidth = Math.max(0.6, 2.0 * (1 - (k / ph.trail.length)));
                        ctx.stroke();
                    }
                }
            });

            // 6. Draw Sink Core Nodes
            sinkTokens.forEach(function (st) {
                var isSel = probeNode && probeNode.type === 'sink' && probeNode.data.id === st.id;
                var baseR = isSel ? 4.8 : (st.id === 0 ? 4.2 : 3.0);

                var sGrad = ctx.createRadialGradient(st.x, st.y, 0, st.x, st.y, baseR * 3.0);
                sGrad.addColorStop(0, isSel ? '#ffffff' : PALETTE.coreHot);
                sGrad.addColorStop(0.4, PALETTE.goldGlow);
                sGrad.addColorStop(1, 'rgba(201, 163, 92, 0)');
                ctx.fillStyle = sGrad;
                ctx.beginPath();
                ctx.arc(st.x, st.y, baseR * 3.0, 0, Math.PI * 2);
                ctx.fill();

                ctx.beginPath();
                ctx.arc(st.x, st.y, baseR, 0, Math.PI * 2);
                ctx.fillStyle = isSel ? '#ffffff' : PALETTE.gold;
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 0.8;
                ctx.stroke();

                if (isSel) {
                    ctx.beginPath();
                    ctx.arc(st.x, st.y, 12, 0, Math.PI * 2);
                    ctx.strokeStyle = PALETTE.gold;
                    ctx.lineWidth = 1.0;
                    ctx.stroke();
                }
            });

            // 7. Draw Sequence Phasor Nodes
            phasors.forEach(function (ph) {
                var isSel = probeNode && probeNode.type === 'phasor' && probeNode.data.id === ph.id;
                var baseR = isSel ? 4.2 : (ph.isSink ? 3.0 : 2.4);
                var extraGlow = ph.glow * 3.5;

                var pCol = ph.r < 0.48 ? PALETTE.olive : PALETTE.terracotta;
                var pGlow = ph.r < 0.48 ? PALETTE.oliveGlow : PALETTE.terracottaGlow;

                var pGrad = ctx.createRadialGradient(ph.x, ph.y, 0, ph.x, ph.y, baseR * 3.0 + extraGlow);
                pGrad.addColorStop(0, isSel ? '#ffffff' : pGlow);
                pGrad.addColorStop(0.5, ph.r < 0.48 ? PALETTE.oliveTint : PALETTE.terracottaTint);
                pGrad.addColorStop(1, 'rgba(0,0,0,0)');
                ctx.fillStyle = pGrad;
                ctx.beginPath();
                ctx.arc(ph.x, ph.y, baseR * 3.0 + extraGlow, 0, Math.PI * 2);
                ctx.fill();

                ctx.beginPath();
                ctx.arc(ph.x, ph.y, baseR, 0, Math.PI * 2);
                ctx.fillStyle = isSel ? '#ffffff' : pCol;
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 0.7;
                ctx.stroke();

                if (isSel) {
                    ctx.beginPath();
                    ctx.arc(ph.x, ph.y, 11, 0, Math.PI * 2);
                    ctx.strokeStyle = PALETTE.gold;
                    ctx.lineWidth = 1.0;
                    ctx.stroke();
                }
            });

            // 8. Draw Active Query Head Indicator
            ctx.beginPath();
            ctx.arc(qX, qY, 6.0, 0, Math.PI * 2);
            ctx.fillStyle = '#ffffff';
            ctx.fill();
            ctx.strokeStyle = PALETTE.gold;
            ctx.lineWidth = 2.0;
            ctx.stroke();

            ctx.fillStyle = PALETTE.coreHot;
            ctx.font = 'bold 9px "JetBrains Mono", monospace';
            ctx.fillText('QUERY q_t', qX + 9, qY + 3);

            ctx.restore();
        }

        // Mode 2: YaRN 128K Multi-Frequency Rotary Phasor Spectrum
        function drawYarnPhaseDisk(t, dt) {
            ctx.save();

            // Wavelength dispersion rings
            var freqRings = [0.28, 0.48, 0.68, 0.88];
            freqRings.forEach(function (fr, idx) {
                var r = maxR * fr;
                ctx.beginPath();
                ctx.arc(cx, cy, r, 0, Math.PI * 2);
                ctx.strokeStyle = idx < 2 ? PALETTE.oliveTint : PALETTE.terracottaTint;
                ctx.lineWidth = 0.8;
                ctx.stroke();
            });

            // 32 Rotary Phasor pairs (YaRN frequency spectrum)
            var closestNode = null;
            var closestDist = 99999;

            for (var i = 0; i < 32; i++) {
                var frac = i / 31;
                var r = maxR * (0.24 + 0.72 * Math.pow(frac, 0.85));
                var isHighFreq = frac < 0.35;
                var isRamp = frac >= 0.35 && frac < 0.65;
                var isLowFreq = frac >= 0.65;

                // YaRN scaling factor
                var yarnFactor = isHighFreq ? 1.0 : (isRamp ? 1.0 + 31.0 * ((frac - 0.35) / 0.3) : 32.0);
                var speed = (2.2 / (1.0 + frac * 2.5)) / (isLowFreq ? 2.5 : 1.0);
                var angPlus = (i * 1.618033 + t * speed) % (Math.PI * 2);
                var angMinus = (-i * 1.618033 - t * speed) % (Math.PI * 2);

                var xP = cx + r * Math.cos(angPlus);
                var yP = cy - r * Math.sin(angPlus);
                var xM = cx + r * Math.cos(angMinus);
                var yM = cy - r * Math.sin(angMinus);

                // Harmonic chord linking conjugate pairs
                ctx.beginPath();
                ctx.moveTo(xP, yP);
                ctx.lineTo(xM, yM);
                ctx.strokeStyle = isHighFreq ? 'rgba(154, 148, 64, 0.22)' : (isRamp ? 'rgba(201, 163, 92, 0.28)' : 'rgba(224, 122, 63, 0.24)');
                ctx.lineWidth = 0.8;
                ctx.stroke();

                // Draw Nodes
                [ {x: xP, y: yP}, {x: xM, y: yM} ].forEach(function (pt, pIdx) {
                    var isPruned = (i % 4 !== 0); // 25% active RoPE
                    var nodeCol = isHighFreq ? PALETTE.olive : (isRamp ? PALETTE.gold : PALETTE.terracotta);
                    var nodeR = isPruned ? 2.2 : 3.4;

                    ctx.beginPath();
                    ctx.arc(pt.x, pt.y, nodeR, 0, Math.PI * 2);
                    ctx.fillStyle = nodeCol;
                    ctx.fill();
                    ctx.strokeStyle = '#ffffff';
                    ctx.lineWidth = 0.6;
                    ctx.stroke();

                    if (isHovered) {
                        var d = Math.hypot(mouseX - pt.x, mouseY - pt.y);
                        if (d < closestDist && d < 35) {
                            closestDist = d;
                            closestNode = {
                                type: 'yarn',
                                data: {
                                    dim: i * 3,
                                    scale: yarnFactor,
                                    isPruned: isPruned,
                                    freq: (1.0 / Math.pow(100000, (i * 3) / 96)).toExponential(3),
                                    x: pt.x, y: pt.y
                                }
                            };
                        }
                    }
                });
            }

            probeNode = closestNode;
            if (probeNode && probeNode.type === 'yarn') {
                var nd = probeNode.data;
                ctx.beginPath();
                ctx.arc(nd.x, nd.y, 11, 0, Math.PI * 2);
                ctx.strokeStyle = PALETTE.gold;
                ctx.lineWidth = 1.2;
                ctx.stroke();
            }

            ctx.restore();
        }

        // Mode 3: Attention Gradient Flow Field
        function drawAttentionVectorField(t) {
            ctx.save();
            var step = 26;
            var arrowLen = 10;
            var cols = Math.floor(width / step);
            var rows = Math.floor(height / step);

            for (var i = 0; i <= cols; i++) {
                for (var j = 0; j <= rows; j++) {
                    var px = i * step + (step / 2);
                    var py = j * step + (step / 2);
                    var dx = px - cx;
                    var dy = py - cy;
                    var dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist > maxR * 1.08 || dist < 10) continue;

                    // Combined Gradient: Inward Sink Attraction + Sliding Window Circulation
                    var sinkPull = Math.min(1.0, 35 / (dist + 5));
                    var u = -0.55 * (dx / dist) * sinkPull - 0.75 * (dy / dist);
                    var v = -0.55 * (dy / dist) * sinkPull + 0.75 * (dx / dist);

                    var ang = Math.atan2(v, u);
                    var flowIntensity = 0.3 + 0.7 * Math.sin(dist * 0.035 - t * 2.2);

                    ctx.beginPath();
                    ctx.moveTo(px, py);
                    ctx.lineTo(px + Math.cos(ang) * arrowLen * flowIntensity, py + Math.sin(ang) * arrowLen * flowIntensity);
                    ctx.strokeStyle = dist < maxR * 0.25 ? 'rgba(201, 163, 92, 0.55)' :
                                     (dist < maxR * 0.60 ? 'rgba(154, 148, 64, 0.40)' : 'rgba(224, 122, 63, 0.30)');
                    ctx.lineWidth = 1.0;
                    ctx.stroke();

                    ctx.beginPath();
                    ctx.arc(px, py, 1.0, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(201, 163, 92, 0.35)';
                    ctx.fill();
                }
            }
            ctx.restore();
        }

        // Draw Shockwave Waves
        function drawPulseWaves(dt) {
            if (pulseWaves.length === 0) return;
            ctx.save();

            for (var i = pulseWaves.length - 1; i >= 0; i--) {
                var wave = pulseWaves[i];
                if (!reduced) {
                    wave.r += wave.speed * dt;
                    wave.alpha = Math.max(0, 1.0 - (wave.r / wave.maxR));
                }

                if (wave.alpha <= 0.01 || wave.r >= wave.maxR) {
                    pulseWaves.splice(i, 1);
                    continue;
                }

                ctx.beginPath();
                ctx.arc(wave.x, wave.y, wave.r, 0, Math.PI * 2);
                ctx.strokeStyle = 'rgba(201, 163, 92, ' + (wave.alpha * 0.65).toFixed(3) + ')';
                ctx.lineWidth = Math.max(1, 3.5 * wave.alpha);
                ctx.stroke();
            }

            ctx.restore();
        }

        // Update Probe HUD Card
        function updateProbeHUD() {
            if (!probeEl || !probeTag || !probeCoords || !probeDecay) return;

            if (isHovered && probeNode) {
                probeEl.style.opacity = '1';
                var targetX = 0, targetY = 0;

                if (probeNode.type === 'sink') {
                    var st = probeNode.data;
                    targetX = st.x; targetY = st.y;
                    probeTag.textContent = st.label;
                    probeCoords.textContent = 'Sink Bias β = +' + st.bias.toFixed(2) + ' · Weight = ' + st.weight.toFixed(3);
                    probeDecay.textContent = 'Softmax Mass Anchor · Clamped [-10, +15]';
                } else if (probeNode.type === 'phasor') {
                    var ph = probeNode.data;
                    targetX = ph.x; targetY = ph.y;
                    probeTag.textContent = 'TOKEN #' + ph.id + ' (CTX ~' + ph.tokenIdx.toLocaleString() + ')';
                    var inSWA = ph.r < 0.48;
                    probeCoords.textContent = inSWA ? 'SWA Window (W=128) · O(1) Ring Buffer' : 'YaRN Extrapolated Manifold';
                    probeDecay.textContent = 'YaRN Scale s = ' + ph.yarnScale.toFixed(2) + ' · ' + (ph.isPruned ? 'Global Unpruned' : 'Pruned 25% RoPE');
                } else if (probeNode.type === 'yarn') {
                    var yd = probeNode.data;
                    targetX = yd.x; targetY = yd.y;
                    probeTag.textContent = 'ROPE DIM #' + yd.dim + ' / 96';
                    probeCoords.textContent = 'Scale Factor s = ' + yd.scale.toFixed(2) + ' · θ = 100K';
                    probeDecay.textContent = 'Base Freq ω_0 = ' + yd.freq + ' · ' + (yd.isPruned ? 'Pruned Global Subspace' : 'Active 25%');
                }

                // Place probe tooltip near target
                var containerRect = container.getBoundingClientRect();
                var cardX = targetX + 16;
                var cardY = targetY - 24;
                if (cardX + 220 > containerRect.width) cardX = targetX - 230;
                if (cardY < 10) cardY = 10;
                probeEl.style.left = cardX + 'px';
                probeEl.style.top = cardY + 'px';
            } else {
                probeEl.style.opacity = '0';
            }
        }

        function renderFrame(dt, force) {
            if (!force && isPaused) return;

            simTime += dt;
            ctx.clearRect(0, 0, width, height);

            drawBackground();
            drawSpiralManifolds(simTime);

            if (currentMode === 'sink') {
                drawSinkGravityWell(simTime, dt);
            } else if (currentMode === 'yarn') {
                drawYarnPhaseDisk(simTime, dt);
            } else if (currentMode === 'field') {
                drawAttentionVectorField(simTime);
            }

            drawPulseWaves(dt);
            updateProbeHUD();

            if (normValEl) normValEl.textContent = '2.00×';
        }

        function loop(ts) {
            var dt = lastTime ? (ts - lastTime) / 1000 : 0.016;
            lastTime = ts;
            if (dt > 0.1) dt = 0.016;

            renderFrame(dt, false);

            if (!isPaused && !reduced) {
                animId = requestAnimationFrame(loop);
            }
        }

        if (!reduced) {
            animId = requestAnimationFrame(loop);
        } else {
            renderFrame(0, true);
        }
    }
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    // 2. Living Pipeline Pass Telemetry (FIG · A1)
    //    Interactive high-DPI Canvas simulation of one full training step:
    //    Forward Activation Pass → Chunked Cross-Entropy Loss →
    //    Autograd Backward Pass (Gradient Checkpointing Activation
    //    Recomputation) → AdamW Parameter Update Step.
    // ------------------------------------------------------------------
    function initPassDiagram() {
        var canvas = document.getElementById('passDiagramCanvas');
        if (!canvas) return;

        var ctx = canvas.getContext('2d');
        if (!ctx) return;

        var container = canvas.parentElement;
        var tooltipEl = document.getElementById('passStageTooltip');
        var stTag = document.getElementById('stTag');
        var stOp = document.getElementById('stOp');
        var stShape = document.getElementById('stShape');
        var stDesc = document.getElementById('stDesc');
        var phaseEl = document.getElementById('phCurrentPhase');
        var tickerText = document.getElementById('passTickerText');
        var tickerBeacon = document.getElementById('passTickerBeacon');
        var pauseBtn = document.getElementById('passPauseBtn');
        var phaseBtns = document.querySelectorAll('.pass-controls .pass-btn[data-phase]');

        var currentPhaseMode = 'cycle';
        var isPaused = false;
        var isHovered = false;
        var mouseX = -9999, mouseY = -9999;
        var hoveredStation = null;
        var animId = null;
        var lastTime = 0;
        var simTime = 0;

        var PALETTE = {
            paperBg: '#0e0c0a',
            paperStation: '#161310',
            paperStationHover: '#1c1813',
            rule: '#2c261c',
            ruleStrong: '#3a3226',
            terracotta: '#e07a3f',
            terracottaGlow: 'rgba(224, 122, 63, 0.75)',
            terracottaTint: 'rgba(224, 122, 63, 0.16)',
            olive: '#9a9440',
            oliveGlow: 'rgba(154, 148, 64, 0.75)',
            oliveTint: 'rgba(154, 148, 64, 0.16)',
            gold: '#c9a35c',
            goldGlow: 'rgba(201, 163, 92, 0.8)',
            goldTint: 'rgba(201, 163, 92, 0.2)',
            ink: '#d8ccb4',
            inkSoft: '#b3a68c',
            inkFaint: '#7a7160'
        };

        var STAGES = [
            {
                id: 1,
                tag: 'STAGE 01 · EMBEDDING',
                badge: '01 · EMB',
                title: 'Embedding',
                sub: 'x_t → 768',
                chip: 'Tied Weights',
                op: 'h_0 = Embedding(x_t) · Tied with LM Head',
                shape: 'Input: [B, 4096] uint32 → Output: [B, 4096, 768] bf16',
                desc: 'TikToken BPE vocab (128,000) · Tied embedding/head parameter weight matrix',
                isParam: true
            },
            {
                id: 2,
                tag: 'STAGE 02 · SLIDING-WINDOW ATTN',
                badge: '02 · SWA',
                title: 'Sliding-Window',
                sub: 'W=128 · GQA',
                chip: 'Sink Bias +4.18',
                op: 'A[q,k] = softmax(q·k/√d + β_sink) · mask_{|q-k|≤128}',
                shape: 'Attention: [B, 8Q/4KV, 4096, 4096] bf16 · Cache W=128 (Ring)',
                desc: '6 SWA layers in alternation · Learned per-head sink bias [-10, +15] · Flash SDPA',
                isParam: true
            },
            {
                id: 3,
                tag: 'STAGE 03 · GLOBAL YaRN ATTN',
                badge: '03 · YARN',
                title: 'Global YaRN',
                sub: 'θ=100K · 128K',
                chip: '25% Pruned RoPE',
                op: 'A[q,k] = softmax(q·k/√d · YaRN(θ=100K, scale=32))',
                shape: 'Full Attn: [B, 8Q/4KV, 4096, 4096] bf16 · Target 128K context',
                desc: '6 Full-Attn layers · Pruned RoPE on 25% head dims · NTK-by-parts extrapolation',
                isParam: true
            },
            {
                id: 4,
                tag: 'STAGE 04 · MoE ROUTER',
                badge: '04 · MOE',
                title: 'MoE Router',
                sub: 'Top-2 / 8 Exp',
                chip: 'Standard Aux α=0.01',
                op: 'g = softmax_{FP32}(W_g · x), top-2 indices ∈ {0..7}',
                shape: 'Gating: [B, 4096, 8] fp32 probs · Router Top-2 index dispatch',
                desc: 'Standard aux load-balancing loss (α=0.01) · FP32 router softmax avoids underflow',
                isParam: true
            },
            {
                id: 5,
                tag: 'STAGE 05 · TRITON GROUPED-GEMM',
                badge: '05 · TRITON',
                title: 'Triton GEMM',
                sub: 'Fused W1/W3',
                chip: 'sm_75 Grouped',
                op: 'y = (silu(W1 · x) ⊙ (W3 · x)) · W2 + SharedExpert(x)',
                shape: 'SwiGLU: [B, 4096, 1536] activations · 1 Shared + 8 Routed',
                desc: 'Fused W1/W3+silu grouped-GEMM (BLOCK_T=16, BLOCK_M=32) · opt-in via moe_dispatch',
                isParam: true
            },
            {
                id: 6,
                tag: 'STAGE 06 · TIED HEAD & LOSS',
                badge: '06 · LOSS',
                title: 'Head & Loss',
                sub: 'Chunked CE',
                chip: 'FP32 Master',
                op: 'ℓ = ∑_chunks CE(logits, target); θ ← θ − η · AdamW(∇θ)',
                shape: 'Logits: [B, 4096, 128000] in chunks of 8192 → Loss scalar ℓ',
                desc: 'Chunked cross-entropy saves VRAM · Fused AdamW FP32 master weights',
                isParam: true
            }
        ];

        var forwardParticles = [];
        for (var f = 0; f < 24; f++) {
            forwardParticles.push({
                xFrac: f / 24,
                speed: 0.18 + 0.08 * Math.random(),
                size: 1.6 + 1.0 * Math.random(),
                lane: (f % 3) - 1
            });
        }

        var backwardParticles = [];
        for (var b = 0; b < 24; b++) {
            backwardParticles.push({
                xFrac: b / 24,
                speed: 0.20 + 0.08 * Math.random(),
                size: 1.6 + 1.0 * Math.random(),
                lane: (b % 3) - 1
            });
        }

        var adamParticles = [];
        var width = 0, height = 0;
        var stations = [];

        function layoutStations() {
            var rect = container.getBoundingClientRect();
            var dpr = Math.min(window.devicePixelRatio || 1, 2);
            width = rect.width;
            height = rect.height || 260;
            canvas.width = Math.round(width * dpr);
            canvas.height = Math.round(height * dpr);
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.scale(dpr, dpr);

            stations = [];
            var numStages = STAGES.length;
            var marginX = 16;
            var availW = width - (marginX * 2);
            var gap = Math.max(8, Math.min(16, (availW - (numStages * 82)) / (numStages - 1)));
            var stationW = Math.max(76, (availW - (gap * (numStages - 1))) / numStages);
            var stationH = Math.min(124, height * 0.52);
            var stationY = (height - stationH) / 2 + 2;

            for (var i = 0; i < numStages; i++) {
                var sx = marginX + i * (stationW + gap);
                stations.push({
                    stage: STAGES[i],
                    x: sx,
                    y: stationY,
                    w: stationW,
                    h: stationH,
                    cx: sx + stationW / 2,
                    cy: stationY + stationH / 2,
                    glowForward: 0,
                    glowBackward: 0,
                    glowAdam: 0
                });
            }
        }

        if (window.ResizeObserver) {
            var ro = new ResizeObserver(function () {
                layoutStations();
                if (reduced || isPaused) render(0, true);
            });
            ro.observe(container);
        } else {
            window.addEventListener('resize', layoutStations);
        }
        layoutStations();

        // Mouse Listeners
        container.addEventListener('mousemove', function (e) {
            var rect = canvas.getBoundingClientRect();
            mouseX = e.clientX - rect.left;
            mouseY = e.clientY - rect.top;
            isHovered = true;
        });

        container.addEventListener('mouseleave', function () {
            isHovered = false;
            mouseX = -9999;
            mouseY = -9999;
            hoveredStation = null;
            if (tooltipEl) tooltipEl.style.opacity = '0';
        });

        container.addEventListener('click', function (e) {
            var rect = canvas.getBoundingClientRect();
            var clickX = e.clientX - rect.left;
            var clickY = e.clientY - rect.top;

            stations.forEach(function (st) {
                if (clickX >= st.x && clickX <= st.x + st.w && clickY >= st.y && clickY <= st.y + st.h) {
                    st.glowForward = 1.0;
                    st.glowBackward = 1.0;
                    for (var k = 0; k < 12; k++) {
                        adamParticles.push({
                            x: st.cx,
                            y: st.cy,
                            vx: (Math.random() - 0.5) * 160,
                            vy: (Math.random() - 0.5) * 160,
                            life: 1.0,
                            color: PALETTE.gold
                        });
                    }
                }
            });
        });

        if (pauseBtn) {
            pauseBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                isPaused = !isPaused;
                pauseBtn.innerHTML = isPaused ? '&#9658;' : '&#10074;&#10074;';
                pauseBtn.setAttribute('aria-label', isPaused ? 'Resume pipeline' : 'Pause pipeline');
                if (!isPaused && !animId) {
                    lastTime = performance.now();
                    loop(lastTime);
                }
            });
        }

        phaseBtns.forEach(function (btn) {
            btn.addEventListener('click', function (e) {
                e.stopPropagation();
                phaseBtns.forEach(function (b) { b.classList.remove('active'); });
                btn.classList.add('active');
                currentPhaseMode = btn.getAttribute('data-phase') || 'cycle';
                if (reduced || isPaused) render(0, true);
            });
        });

        var CYCLE_DURATION = 8.6;

        function getCycleState(t) {
            if (currentPhaseMode === 'forward') {
                return { phase: 'forward', progress: (t * 0.4) % 1.0, subText: 'FORWARD ACTIVATIONS STREAMING' };
            }
            if (currentPhaseMode === 'backward') {
                return { phase: 'backward', progress: (t * 0.4) % 1.0, subText: 'AUTOGRAD GRADIENT PROPAGATION & CHECKPOINTING' };
            }

            var cycleT = t % CYCLE_DURATION;
            if (cycleT < 3.2) {
                return { phase: 'forward', progress: cycleT / 3.2, subText: 'FORWARD · Activations x_t → 12 Layers (SWA ↔ Full) → MoE Grouped-GEMM' };
            } else if (cycleT < 4.2) {
                return { phase: 'loss', progress: (cycleT - 3.2) / 1.0, subText: 'LOSS COMPUTATION · Chunked Cross-Entropy (chunk=8192)' };
            } else if (cycleT < 7.0) {
                return { phase: 'backward', progress: (cycleT - 4.2) / 2.8, subText: 'AUTOGRAD BACKWARD · Checkpointing Recomputes Activations Every 3rd Layer' };
            } else if (cycleT < 8.0) {
                return { phase: 'adam', progress: (cycleT - 7.0) / 1.0, subText: 'ADAMW STEP · Fused FP32 Master Weight Updates θ ← θ - η·∇ℓ' };
            } else {
                return { phase: 'rest', progress: (cycleT - 8.0) / 0.6, subText: 'STEP COMMITTED · Next Mini-Batch Ingest' };
            }
        }

        function drawBackground() {
            ctx.fillStyle = PALETTE.paperBg;
            ctx.fillRect(0, 0, width, height);

            ctx.save();
            // Subtle technical grid
            ctx.strokeStyle = 'rgba(58, 50, 38, 0.22)';
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 8]);
            for (var x = 20; x < width; x += 40) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
                ctx.stroke();
            }
            ctx.setLineDash([]);

            // Upper Forward Conduit
            var fwdY = height * 0.18;
            ctx.beginPath();
            ctx.moveTo(14, fwdY);
            ctx.lineTo(width - 14, fwdY);
            ctx.strokeStyle = 'rgba(224, 122, 63, 0.28)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 6]);
            ctx.stroke();

            // Lower Backward Conduit
            var bwdY = height * 0.82;
            ctx.beginPath();
            ctx.moveTo(14, bwdY);
            ctx.lineTo(width - 14, bwdY);
            ctx.strokeStyle = 'rgba(154, 148, 64, 0.28)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 6]);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.fillStyle = PALETTE.terracotta;
            ctx.font = 'bold 8.5px "JetBrains Mono", monospace';
            ctx.textAlign = 'left';
            ctx.fillText('FORWARD CONDUIT (ACTIVATIONS) →', 20, fwdY - 6);

            ctx.fillStyle = PALETTE.olive;
            ctx.font = 'bold 8.5px "JetBrains Mono", monospace';
            ctx.textAlign = 'right';
            ctx.fillText('← BACKWARD CONDUIT (GRADIENTS)', width - 20, bwdY + 14);
            ctx.restore();
        }

        function drawParticles(state, dt) {
            ctx.save();
            var fwdY = height * 0.18;
            var bwdY = height * 0.82;

            // Forward streaming particles
            if (state.phase === 'forward' || state.phase === 'loss' || currentPhaseMode === 'forward') {
                forwardParticles.forEach(function (p) {
                    if (!reduced) p.xFrac += p.speed * dt;
                    if (p.xFrac > 1.0) p.xFrac = 0.0;

                    var px = 16 + p.xFrac * (width - 32);
                    var py = fwdY + p.lane * 3.0;

                    ctx.beginPath();
                    ctx.arc(px, py, p.size, 0, Math.PI * 2);
                    ctx.fillStyle = PALETTE.terracottaGlow;
                    ctx.fill();

                    // Forward motion trail
                    ctx.beginPath();
                    ctx.moveTo(px, py);
                    ctx.lineTo(px - 14 * p.speed, py);
                    ctx.strokeStyle = 'rgba(224, 122, 63, 0.35)';
                    ctx.lineWidth = p.size * 0.7;
                    ctx.stroke();
                });
            }

            // Backward streaming particles
            if (state.phase === 'backward' || currentPhaseMode === 'backward') {
                backwardParticles.forEach(function (p) {
                    if (!reduced) p.xFrac += p.speed * dt;
                    if (p.xFrac > 1.0) p.xFrac = 0.0;

                    var px = width - 16 - p.xFrac * (width - 32);
                    var py = bwdY + p.lane * 3.0;

                    ctx.beginPath();
                    ctx.arc(px, py, p.size, 0, Math.PI * 2);
                    ctx.fillStyle = PALETTE.oliveGlow;
                    ctx.fill();

                    // Backward motion trail
                    ctx.beginPath();
                    ctx.moveTo(px, py);
                    ctx.lineTo(px + 14 * p.speed, py);
                    ctx.strokeStyle = 'rgba(154, 148, 64, 0.35)';
                    ctx.lineWidth = p.size * 0.7;
                    ctx.stroke();
                });
            }

            // AdamW Burst Particles
            for (var k = adamParticles.length - 1; k >= 0; k--) {
                var ap = adamParticles[k];
                ap.x += ap.vx * dt;
                ap.y += ap.vy * dt;
                ap.life -= dt * 1.6;

                if (ap.life <= 0) {
                    adamParticles.splice(k, 1);
                    continue;
                }

                ctx.beginPath();
                ctx.arc(ap.x, ap.y, 2.0 * ap.life, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(201, 163, 92, ' + ap.life.toFixed(2) + ')';
                ctx.fill();
            }

            ctx.restore();
        }

        function drawStations(state, dt) {
            ctx.save();
            var closest = null;
            var numStations = stations.length;

            stations.forEach(function (st, idx) {
                var frac = idx / (numStations - 1);

                // Compute activation level
                var isFwdActive = false;
                var isBwdActive = false;
                var isAdamActive = (state.phase === 'adam');

                if (state.phase === 'forward') {
                    isFwdActive = state.progress >= (frac - 0.12) && state.progress <= (frac + 0.18);
                } else if (state.phase === 'loss') {
                    isFwdActive = (idx === numStations - 1);
                } else if (state.phase === 'backward') {
                    var bwdFrac = 1.0 - frac;
                    isBwdActive = state.progress >= (bwdFrac - 0.12) && state.progress <= (bwdFrac + 0.18);
                }

                if (isFwdActive) st.glowForward = 1.0;
                else st.glowForward = Math.max(0, st.glowForward - dt * 2.2);

                if (isBwdActive) st.glowBackward = 1.0;
                else st.glowBackward = Math.max(0, st.glowBackward - dt * 2.2);

                if (isAdamActive) st.glowAdam = 1.0;
                else st.glowAdam = Math.max(0, st.glowAdam - dt * 2.0);

                // Check Hover
                var isHover = isHovered && (mouseX >= st.x && mouseX <= st.x + st.w && mouseY >= st.y && mouseY <= st.y + st.h);
                if (isHover) closest = st;

                // Station Box Drawing
                var cardBg = isHover ? PALETTE.paperStationHover : PALETTE.paperStation;
                var borderColor = PALETTE.ruleStrong;

                if (st.glowAdam > 0.1) {
                    borderColor = PALETTE.gold;
                    cardBg = 'rgba(201, 163, 92, ' + (0.15 * st.glowAdam).toFixed(2) + ')';
                } else if (st.glowForward > 0.1) {
                    borderColor = PALETTE.terracotta;
                    cardBg = 'rgba(224, 122, 63, ' + (0.14 * st.glowForward).toFixed(2) + ')';
                } else if (st.glowBackward > 0.1) {
                    borderColor = PALETTE.olive;
                    cardBg = 'rgba(154, 148, 64, ' + (0.14 * st.glowBackward).toFixed(2) + ')';
                }

                // Station Background & Border
                ctx.fillStyle = cardBg;
                ctx.strokeStyle = isHover ? PALETTE.gold : borderColor;
                ctx.lineWidth = (isHover || st.glowForward > 0.3 || st.glowBackward > 0.3) ? 1.5 : 1.0;

                ctx.beginPath();
                ctx.rect(st.x, st.y, st.w, st.h);
                ctx.fill();
                ctx.stroke();

                // Corner Technical Brackets
                var crLen = 4;
                ctx.strokeStyle = isHover ? PALETTE.gold : PALETTE.ruleStrong;
                ctx.lineWidth = 1;
                ctx.beginPath();
                // Top-Left
                ctx.moveTo(st.x, st.y + crLen); ctx.lineTo(st.x, st.y); ctx.lineTo(st.x + crLen, st.y);
                // Top-Right
                ctx.moveTo(st.x + st.w - crLen, st.y); ctx.lineTo(st.x + st.w, st.y); ctx.lineTo(st.x + st.w, st.y + crLen);
                // Bottom-Left
                ctx.moveTo(st.x, st.y + st.h - crLen); ctx.lineTo(st.x, st.y + st.h); ctx.lineTo(st.x + crLen, st.y + st.h);
                // Bottom-Right
                ctx.moveTo(st.x + st.w - crLen, st.y + st.h); ctx.lineTo(st.x + st.w, st.y + st.h); ctx.lineTo(st.x + st.w, st.y + st.h - crLen);
                ctx.stroke();

                // Status Beacon LED Dot
                var ledColor = PALETTE.inkFaint;
                if (st.glowAdam > 0.2) ledColor = PALETTE.gold;
                else if (st.glowForward > 0.2) ledColor = PALETTE.terracotta;
                else if (st.glowBackward > 0.2) ledColor = PALETTE.olive;

                ctx.beginPath();
                ctx.arc(st.x + 8, st.y + 10, 2.5, 0, Math.PI * 2);
                ctx.fillStyle = ledColor;
                ctx.fill();

                // Responsive title & badge for compact screens
                var isCompact = st.w < 70;
                var displayBadge = isCompact ? '0' + st.stage.id : st.stage.badge;
                var displayTitle = isCompact ? (st.stage.id === 1 ? 'Embed' : st.stage.id === 2 ? 'SWA' : st.stage.id === 3 ? 'YaRN' : st.stage.id === 4 ? 'MoE' : st.stage.id === 5 ? 'Triton' : 'Loss') : st.stage.title;

                // Stage Number Badge
                ctx.font = isCompact ? 'bold 7.5px "JetBrains Mono", monospace' : 'bold 8.5px "JetBrains Mono", monospace';
                ctx.fillStyle = (st.glowForward > 0.2 ? PALETTE.terracotta : (st.glowBackward > 0.2 ? PALETTE.olive : PALETTE.inkSoft));
                ctx.textAlign = 'left';
                ctx.fillText(displayBadge, st.x + (isCompact ? 13 : 14), st.y + 13);

                // Stage Title
                ctx.font = isCompact ? 'bold 9px "JetBrains Mono", monospace' : 'bold 10px "JetBrains Mono", monospace';
                ctx.fillStyle = isHover ? '#ffffff' : PALETTE.ink;
                ctx.textAlign = 'center';
                ctx.fillText(displayTitle, st.cx, st.y + st.h * 0.40);

                // Subtitle
                if (st.h > 80) {
                    ctx.font = isCompact ? '7.5px "JetBrains Mono", monospace' : '8.5px "JetBrains Mono", monospace';
                    ctx.fillStyle = PALETTE.gold;
                    ctx.fillText(st.stage.sub, st.cx, st.y + st.h * 0.60);
                }

                // Chip Tag
                if (st.h > 100 && !isCompact) {
                    ctx.font = '8px "JetBrains Mono", monospace';
                    ctx.fillStyle = PALETTE.inkSoft;
                    ctx.fillText(st.stage.chip, st.cx, st.y + st.h * 0.78);
                }

                // Re-compute / Checkpoint Tag during Backward
                if (st.glowBackward > 0.3 && (idx >= 1 && idx <= 4)) {
                    ctx.fillStyle = PALETTE.olive;
                    ctx.font = 'bold 7.5px "JetBrains Mono", monospace';
                    ctx.fillText('RE-COMPUTE', st.cx, st.y + st.h - 6);
                } else if (st.glowAdam > 0.3) {
                    ctx.fillStyle = PALETTE.gold;
                    ctx.font = 'bold 7.5px "JetBrains Mono", monospace';
                    ctx.fillText('θ UPDATE', st.cx, st.y + st.h - 6);
                }

                // Inter-stage dataflow bridge connector
                if (idx < numStations - 1) {
                    var nextSt = stations[idx + 1];
                    var gapStartX = st.x + st.w;
                    var gapEndX = nextSt.x;
                    var midY = st.cy;

                    ctx.save();
                    ctx.beginPath();
                    ctx.moveTo(gapStartX, midY);
                    ctx.lineTo(gapEndX, midY);
                    ctx.strokeStyle = (st.glowForward > 0.2 || nextSt.glowForward > 0.2) ? 'rgba(224, 122, 63, 0.7)' :
                                      (st.glowBackward > 0.2 || nextSt.glowBackward > 0.2) ? 'rgba(154, 148, 64, 0.7)' :
                                      PALETTE.ruleStrong;
                    ctx.lineWidth = (st.glowForward > 0.2 || nextSt.glowForward > 0.2 || st.glowBackward > 0.2 || nextSt.glowBackward > 0.2) ? 1.5 : 1.0;
                    ctx.stroke();

                    // Small directional chevron arrow
                    var arrowX = (gapStartX + gapEndX) / 2;
                    ctx.beginPath();
                    if (state.phase === 'backward') {
                        ctx.moveTo(arrowX + 2.5, midY - 2.5);
                        ctx.lineTo(arrowX - 2.5, midY);
                        ctx.lineTo(arrowX + 2.5, midY + 2.5);
                        ctx.strokeStyle = PALETTE.olive;
                    } else {
                        ctx.moveTo(arrowX - 2.5, midY - 2.5);
                        ctx.lineTo(arrowX + 2.5, midY);
                        ctx.lineTo(arrowX - 2.5, midY + 2.5);
                        ctx.strokeStyle = (st.glowForward > 0.2) ? PALETTE.terracotta : PALETTE.rule;
                    }
                    ctx.lineWidth = 1;
                    ctx.stroke();
                    ctx.restore();
                }
            });

            hoveredStation = closest;
            ctx.restore();
        }

        function updateHUD(state) {
            if (phaseEl) {
                var phaseName = 'FORWARD ACTIVATIONS';
                if (state.phase === 'loss') phaseName = 'CHUNKED CE LOSS (ℓ)';
                else if (state.phase === 'backward') phaseName = 'AUTOGRAD BACKWARD (RECOMPUTE)';
                else if (state.phase === 'adam') phaseName = 'ADAMW WEIGHT UPDATE (FP32)';
                else if (state.phase === 'rest') phaseName = 'STANDBY / NEXT MINI-BATCH';
                phaseEl.textContent = phaseName;
            }

            if (tickerText) {
                tickerText.textContent = state.subText;
            }

            if (tickerBeacon) {
                tickerBeacon.style.color = (state.phase === 'forward' ? PALETTE.terracotta : (state.phase === 'backward' ? PALETTE.olive : PALETTE.gold));
            }

            if (hoveredStation && tooltipEl) {
                tooltipEl.style.opacity = '1';
                var st = hoveredStation.stage;
                var targetX = hoveredStation.x;
                var targetY = hoveredStation.y;

                var cardW = 320;
                var cardH = 80;

                var leftPx = targetX + hoveredStation.w / 2 - cardW / 2;
                var topPx = targetY - cardH - 12;
                if (topPx < 8) topPx = targetY + hoveredStation.h + 12;

                leftPx = Math.max(12, Math.min(width - cardW - 12, leftPx));

                tooltipEl.style.left = Math.round(leftPx) + 'px';
                tooltipEl.style.top = Math.round(topPx) + 'px';

                if (stTag) stTag.textContent = st.tag;
                if (stOp) stOp.textContent = st.op;
                if (stShape) stShape.textContent = st.shape;
                if (stDesc) stDesc.textContent = st.desc;
            } else if (tooltipEl) {
                tooltipEl.style.opacity = '0';
            }
        }

        function render(dt, force) {
            if (!force && isPaused) return;

            simTime += dt;
            ctx.clearRect(0, 0, width, height);

            var state = getCycleState(simTime);

            drawBackground();
            drawParticles(state, dt);
            drawStations(state, dt);
            updateHUD(state);
        }

        function loop(timestamp) {
            if (!lastTime) lastTime = timestamp;
            var dt = Math.min((timestamp - lastTime) / 1000, 0.1);
            lastTime = timestamp;

            render(dt, false);

            if (!isPaused && !reduced) {
                animId = requestAnimationFrame(loop);
            }
        }

        if (reduced) {
            render(0, true);
        } else {
            lastTime = performance.now();
            animId = requestAnimationFrame(loop);

            document.addEventListener('visibilitychange', function () {
                if (document.hidden) {
                    if (animId) {
                        cancelAnimationFrame(animId);
                        animId = null;
                    }
                } else if (!isPaused && !animId) {
                    lastTime = performance.now();
                    animId = requestAnimationFrame(loop);
                }
            });
        }
    }
    // 3. Expandable Code Blocks (>14 lines) & Copy-to-Clipboard
    // ------------------------------------------------------------------
    function fallbackCopy(text) {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.top = '-9999px';
        ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); } catch (e) {}
        document.body.removeChild(ta);
    }

    window.toggleCode = function (btn) {
        var wrapper = btn.closest('.code-wrapper');
        if (!wrapper) return;
        var collapsed = wrapper.classList.toggle('collapsed');
        var nLines = wrapper.getAttribute('data-lines') || '';
        if (!nLines) {
            var codeEl = wrapper.querySelector('code');
            if (codeEl) {
                var lines = codeEl.innerText.split('\n').length;
                nLines = String(lines);
            }
        }
        btn.textContent = collapsed ? ('expand \u25be \u00b7 ' + nLines + ' lines') : 'collapse \u25b4';
    };

    window.copyCode = function (btn) {
        var wrapper = btn.closest('.code-wrapper');
        if (!wrapper) return;
        var code = wrapper.querySelector('pre code') || wrapper.querySelector('code');
        if (!code) return;
        var text = code.innerText || code.textContent;

        function flash() {
            var orig = btn.textContent;
            btn.textContent = 'Copied!';
            btn.classList.add('copied');
            setTimeout(function () {
                btn.textContent = orig;
                btn.classList.remove('copied');
            }, 1800);
        }

        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(flash).catch(function () {
                fallbackCopy(text);
                flash();
            });
        } else {
            fallbackCopy(text);
            flash();
        }
    };

    // ------------------------------------------------------------------
    // 4. Navigation Filtering & Sidebar Mobile Toggle
    // ------------------------------------------------------------------
    window.filterNav = function () {
        var input = document.getElementById('navSearch');
        if (!input) return;
        var q = (input.value || '').toLowerCase().trim();
        var links = document.querySelectorAll('.nav-link');
        var groupHeaders = document.querySelectorAll('.nav-group');

        links.forEach(function (l) {
            var txt = l.textContent.toLowerCase();
            var item = l.closest('.nav-item');
            if (!item) return;
            item.style.display = (!q || txt.indexOf(q) !== -1) ? '' : 'none';
        });

        // Hide empty category groups when filtered
        groupHeaders.forEach(function (grp) {
            var visibleItems = grp.querySelectorAll('.nav-item:not([style*="display: none"])');
            grp.style.display = (!q || visibleItems.length > 0) ? '' : 'none';
        });
    };
    window.toggleSidebar = function () {
        var sb = document.getElementById('sidebar');
        if (sb) sb.classList.toggle('open');
    };

    // ------------------------------------------------------------------
    // 5. Table of Contents Scrollspy
    // ------------------------------------------------------------------
    function initScrollspy() {
        var tocSidebar = document.querySelector('.toc-sidebar');
        var tocLinks = Array.from(document.querySelectorAll('.toc-link'));
        if (!tocLinks.length) return;

        var headings = [];
        tocLinks.forEach(function (link) {
            var href = link.getAttribute('href');
            if (href && href.startsWith('#')) {
                var targetId = href.substring(1);
                var el = document.getElementById(targetId);
                if (el) headings.push({ el: el, link: link, id: targetId });
            }
        });
        if (!headings.length) return;

        var isTicking = false;
        var activeItem = null;

        function updateSpy() {
            isTicking = false;

            // Reading threshold: 140px below the viewport top (accounts for sticky site-header)
            var threshold = 140;
            var current = null;

            // Find the active heading in viewport DOM order
            for (var i = 0; i < headings.length; i++) {
                var rect = headings[i].el.getBoundingClientRect();
                if (rect.top <= threshold) {
                    current = headings[i];
                } else {
                    break;
                }
            }

            // Default to first heading when near the top of the page
            if (!current && headings.length > 0 && window.scrollY < 300) {
                current = headings[0];
            }

            // If scrolled to the very bottom of the document, activate the last heading
            if ((window.innerHeight + window.scrollY) >= (document.body.offsetHeight - 60)) {
                current = headings[headings.length - 1];
            }

            if (current !== activeItem) {
                activeItem = current;

                tocLinks.forEach(function (l) {
                    l.classList.remove('active');
                });

                if (current && current.link) {
                    current.link.classList.add('active');

                    // Auto-scroll TOC sidebar to keep the active link visible and centered
                    if (tocSidebar) {
                        var sidebarRect = tocSidebar.getBoundingClientRect();
                        var linkRect = current.link.getBoundingClientRect();

                        // Check if link is outside the comfortable visible zone of TOC sidebar
                        if (linkRect.top < sidebarRect.top + 30 || linkRect.bottom > sidebarRect.bottom - 30) {
                            var linkOffsetInSidebar = current.link.offsetTop;
                            var targetScroll = linkOffsetInSidebar - (tocSidebar.clientHeight / 2) + (current.link.clientHeight / 2);
                            tocSidebar.scrollTo({
                                top: Math.max(0, targetScroll),
                                behavior: 'smooth'
                            });
                        }
                    }
                }
            }
        }

        function onScroll() {
            if (!isTicking) {
                requestAnimationFrame(updateSpy);
                isTicking = true;
            }
        }

        window.addEventListener('scroll', onScroll, { passive: true });
        window.addEventListener('resize', onScroll, { passive: true });

        setTimeout(updateSpy, 100);
        setTimeout(updateSpy, 500);
    }

    // ------------------------------------------------------------------
    // 6. Highlight.js & KaTeX bootstrap (also runs on doc pages).
    // ------------------------------------------------------------------
    function initHighlightAndMath() {
        if (window.hljs) hljs.highlightAll();
        if (window.renderMathInElement) {
            renderMathInElement(document.body, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '\\[', right: '\\]', display: true},
                    {left: '\\(', right: '\\)', display: false},
                    {left: '$', right: '$', display: false}
                ],
                ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
                throwOnError: false
            });
        }
    }

    // ------------------------------------------------------------------
    // Boot Initialization
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    // MoE Routing Playground (FIG · C4, lives on docs/concepts/moe.html)
    // Renders 8 expert columns, k=2 glow indicators, and a load-balance aux
    // loss curve in the background; click a column to toggle routing bias.
    // ------------------------------------------------------------------
    function initMoeRouting() {
        var canvas = document.getElementById('moeRoutingCanvas');
        if (!canvas) return;
        var container = canvas.parentElement;
        var ctx0 = canvas.getContext('2d');
        var width = 0, height = 0, dpr = 1;
        var bias = [1, 1, 1, 1, 1, 1, 1, 1];
        function resize() {
            var rect = container.getBoundingClientRect();
            if (rect.width === 0) return;
            width = rect.width; height = rect.height;
            dpr = window.devicePixelRatio || 1;
            canvas.width = Math.floor(width * dpr);
            canvas.height = Math.floor(height * dpr);
            canvas.style.width = width + 'px';
            canvas.style.height = height + 'px';
            ctx0.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        if (window.ResizeObserver) new ResizeObserver(resize).observe(container);
        window.addEventListener('resize', resize);
        resize();

        container.addEventListener('click', function (e) {
            var rect = container.getBoundingClientRect();
            var col = Math.floor((e.clientX - rect.left) / (rect.width / 8));
            if (col >= 0 && col < 8) bias[col] = bias[col] > 0.5 ? 0.2 : 1.4;
        });

        function draw(t) {
            ctx0.clearRect(0, 0, width, height);
            ctx0.fillStyle = '#0e0c0a'; ctx0.fillRect(0, 0, width, height);
            var W = width / 8;
            for (var i = 0; i < 8; i++) {
                var bs = bias[i] + Math.sin(t + i) * 0.05;
                var h = (height - 24) * Math.min(1, bs * 0.75);
                var x = i * W + 6;
                var y = height - 8 - h;
                ctx0.fillStyle = (i % 4 < 2) ? 'rgba(198,99,70,0.78)' : 'rgba(198,158,78,0.78)';
                ctx0.fillRect(x, y, W - 12, h);
                ctx0.fillStyle = 'rgba(232,220,200,0.62)';
                ctx0.font = '9px IBM Plex Mono, monospace';
                ctx0.fillText('E' + i, x, y - 4);
            }
            // Shared expert line at top
            ctx0.strokeStyle = 'rgba(180,162,102,0.65)';
            ctx0.setLineDash([3, 3]);
            ctx0.beginPath();
            ctx0.moveTo(0, 12); ctx0.lineTo(width, 12);
            ctx0.stroke();
            ctx0.setLineDash([]);
            ctx0.fillStyle = 'rgba(180,162,102,0.85)';
            ctx0.fillText('+ SHARED SwiGLU', 6, 10);
        }
        if (!reduced) {
            var t = 0;
            (function loop() {
                t += 0.016;
                draw(t);
                requestAnimationFrame(loop);
            })();
        } else draw(0);
    }

    // ------------------------------------------------------------------
    // Sink-Bias Toggle (FIG · C2, lives on docs/concepts/attention-sinks.html)
    // Visualizes the learned per-head sink bias with the [-10, +15] clamp.
    // ------------------------------------------------------------------
    function initSinkBiasToggle() {
        var canvas = document.getElementById('sinkBiasCanvas');
        if (!canvas) return;
        var container = canvas.parentElement;
        var ctx0 = canvas.getContext('2d');
        var width = 0, height = 0, dpr = 1;
        var biasValEl = document.getElementById('sinkBiasValue');
        function resize() {
            var rect = container.getBoundingClientRect();
            if (rect.width === 0) return;
            width = rect.width; height = rect.height;
            dpr = window.devicePixelRatio || 1;
            canvas.width = Math.floor(width * dpr);
            canvas.height = Math.floor(height * dpr);
            canvas.style.width = width + 'px';
            canvas.style.height = height + 'px';
            ctx0.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        if (window.ResizeObserver) new ResizeObserver(resize).observe(container);
        window.addEventListener('resize', resize);
        resize();

        var dragging = false;
        var bias = 5.2;
        function setBias(v) { bias = Math.max(-10, Math.min(15, v)); }
        function draw(t) {
            ctx0.clearRect(0, 0, width, height);
            ctx0.fillStyle = '#0e0c0a'; ctx0.fillRect(0, 0, width, height);
            // Clamp range guide
            var padX = 36;
            var barY = height / 2;
            ctx0.strokeStyle = 'rgba(232,220,200,0.20)';
            ctx0.lineWidth = 1;
            ctx0.beginPath();
            ctx0.moveTo(padX, barY); ctx0.lineTo(width - padX, barY);
            ctx0.stroke();
            // Tick marks
            for (var v = -10; v <= 15; v += 5) {
                var x = padX + ((v + 10) / 25) * (width - 2 * padX);
                ctx0.strokeStyle = 'rgba(232,220,200,0.18)';
                ctx0.beginPath(); ctx0.moveTo(x, barY - 4); ctx0.lineTo(x, barY + 4); ctx0.stroke();
                ctx0.fillStyle = 'rgba(180,162,102,0.55)';
                ctx0.font = '9px IBM Plex Mono, monospace';
                ctx0.fillText(v.toString(), x - 6, barY + 18);
            }
            // Clamp band
            ctx0.fillStyle = 'rgba(198,99,70,0.10)';
            ctx0.fillRect(padX, barY - 12, (5 / 25) * (width - 2 * padX), 24);
            // Bias head
            var bx = padX + ((bias + 10) / 25) * (width - 2 * padX);
            ctx0.fillStyle = 'rgba(198,158,78,0.95)';
            ctx0.beginPath();
            ctx0.arc(bx, barY, 9, 0, Math.PI * 2);
            ctx0.fill();
            ctx0.fillStyle = '#1a1612';
            ctx0.font = '700 10px IBM Plex Mono, monospace';
            ctx0.fillText(bias.toFixed(2), bx - 14, barY + 4);
            // Sink tokens overlay
            var sinkY = barY - 60;
            for (var i = 0; i < 4; i++) {
                ctx0.fillStyle = 'rgba(198,158,78,' + (0.45 + 0.18 * i).toFixed(3) + ')';
                ctx0.fillRect(padX + i * 18, sinkY, 12, 18);
                ctx0.fillStyle = 'rgba(232,220,200,0.72)';
                ctx0.font = '9px IBM Plex Mono, monospace';
                ctx0.fillText('t' + i, padX + i * 18, sinkY - 4);
            }
            ctx0.fillStyle = 'rgba(180,162,102,0.78)';
            ctx0.fillText('SINK TOKENS', padX + 76, sinkY + 10);

            if (biasValEl) biasValEl.textContent = (bias >= 0 ? '+' : '') + bias.toFixed(3);
        }
        function pickBias(clientX) {
            var rect = container.getBoundingClientRect();
            var padX = 36;
            var ratio = (clientX - rect.left - padX) / (rect.width - 2 * padX);
            setBias(ratio * 25 - 10);
        }
        container.addEventListener('mousedown', function (e) { dragging = true; pickBias(e.clientX); });
        container.addEventListener('mousemove', function (e) { if (dragging) pickBias(e.clientX); });
        container.addEventListener('mouseup', function () { dragging = false; });
        container.addEventListener('mouseleave', function () { dragging = false; });

        if (!reduced) {
            var t = 0;
            (function loop() {
                t += 0.016;
                draw(t);
                requestAnimationFrame(loop);
            })();
        } else draw(0);
    }

    function initMechanismExploders() {
        // Lab 1: SWA KV Cache Calculator
        var swaSlider = document.getElementById('swaContextSlider');
        var swaLabel = document.getElementById('swaContextLabel');
        var statFull = document.getElementById('statFullVram');
        var statSwa = document.getElementById('statSwaVram');
        var statSaved = document.getElementById('statSwaSaved');
        var compareBtn = document.getElementById('swaCompareBtn');

        if (swaSlider && swaLabel) {
            function updateSwaCalc() {
                var L = parseInt(swaSlider.value, 10);
                swaLabel.textContent = L.toLocaleString() + ' tokens';
                // GQA: 12 layers, 2 (K+V), 4 KV heads, 96 head_dim, 2 bytes (bf16)
                var fullBytes = 12 * 2 * L * 4 * 96 * 2;
                var W = 128;
                var globalBytes = 6 * 2 * L * 4 * 96 * 2;
                var swaBytes = 6 * 2 * Math.min(L, W) * 4 * 96 * 2;
                var mixedBytes = globalBytes + swaBytes;
                var fullGB = (fullBytes / (1024 * 1024 * 1024)).toFixed(2);
                var mixedGB = (mixedBytes / (1024 * 1024 * 1024)).toFixed(2);
                var savedGB = ((fullBytes - mixedBytes) / (1024 * 1024 * 1024)).toFixed(2);
                var ratio = (fullBytes / mixedBytes).toFixed(2);
                if (statFull) statFull.textContent = fullGB + ' GB';
                if (statSwa) statSwa.textContent = mixedGB + ' GB';
                if (statSaved) statSaved.textContent = savedGB + ' GB (' + ratio + '\u00d7 cut)';
            }
            swaSlider.addEventListener('input', updateSwaCalc);
            swaSlider.addEventListener('change', updateSwaCalc);
            updateSwaCalc();
        }

        if (compareBtn && swaSlider) {
            compareBtn.addEventListener('click', function () {
                swaSlider.value = 131072;
                swaSlider.dispatchEvent(new Event('input'));
                var origText = compareBtn.textContent;
                compareBtn.textContent = '\u2713 At 128K: O(1) decode per step';
                setTimeout(function () { compareBtn.textContent = origText; }, 2000);
            });
        }

        // Lab 2: Sink Bias Monitor
        var sinkDisplay = document.getElementById('sinkMonitorDisplay');
        var sinkMeanEl = document.getElementById('sinkMeanVal');
        var sinkStatusEl = document.getElementById('sinkStatus');
        var randomizeBtn = document.getElementById('sinkRandomizeBtn');

        function renderSinkBars(biases) {
            if (!sinkDisplay) return;
            var CLAMP_MIN = -10, CLAMP_MAX = 15;
            var html = [], sum = 0;
            for (var h = 0; h < 8; h++) {
                var clamped = Math.max(CLAMP_MIN, Math.min(CLAMP_MAX, biases[h]));
                sum += clamped;
                var pct = Math.max(0, Math.min(100, ((clamped - CLAMP_MIN) / (CLAMP_MAX - CLAMP_MIN)) * 100));
                var color = clamped < 0 ? 'var(--terracotta)' : 'var(--olive)';
                html.push(
                    '<div class="sink-bar-row">' +
                    '<span class="sink-bar-label">Head ' + h + '</span>' +
                    '<div class="sink-bar-track"><div class="sink-bar-fill" style="width:' + pct.toFixed(0) + '%;background:' + color + ';"></div></div>' +
                    '<span class="sink-bar-val">' + (clamped >= 0 ? '+' : '') + clamped.toFixed(2) + '</span></div>');
            }
            sinkDisplay.innerHTML = html.join('');
            if (sinkMeanEl) sinkMeanEl.textContent = (sum / 8 >= 0 ? '+' : '') + (sum / 8).toFixed(2);
            if (sinkStatusEl) {
                var maxB = Math.max.apply(null, biases.map(function (b) { return Math.max(CLAMP_MIN, Math.min(CLAMP_MAX, b)); }));
                sinkStatusEl.textContent = maxB > 3 ? 'ACTIVE \u00b7 TOKENS 0..3' : 'WEAK \u00b7 RE-TRAINING NEEDED';
            }
        }

        if (randomizeBtn) {
            randomizeBtn.addEventListener('click', function () {
                var biases = [];
                for (var i = 0; i < 8; i++) biases.push(-4 + Math.random() * 18);
                renderSinkBars(biases);
                var orig = randomizeBtn.textContent;
                randomizeBtn.textContent = '\u2713 Clamped [\u221210, +15]';
                setTimeout(function () { randomizeBtn.textContent = orig; }, 1500);
            });
        }

        // Lab 3: MoE Router Playground (top-2 of 8 + 1 shared)
        var miniGrid = document.getElementById('moeMiniGrid');
        var activeListEl = document.getElementById('moeActiveList');
        var routeBatchBtn = document.getElementById('moeRouteBatchBtn');
        var dispatchValEl = document.getElementById('moeDispatchVal');

        function updateMoEActiveList() {
            if (!miniGrid || !activeListEl) return;
            var chosen = [];
            miniGrid.querySelectorAll('.moe-mini-cell.active:not(.shared)').forEach(function (c) {
                var n = parseInt(c.getAttribute('data-exp'), 10);
                if (!isNaN(n)) chosen.push(n);
            });
            chosen.sort(function (a, b) { return a - b; });
            activeListEl.textContent = (chosen.length ? chosen.map(function (i) {
                return '#' + (i < 10 ? '0' : '') + i;
            }).join(', ') : 'None') + ' + shared';
        }

        if (miniGrid) {
            var html = [];
            for (var i = 0; i < 8; i++) {
                var pad = '0' + i;
                html.push('<div class="moe-mini-cell' + (i === 2 || i === 5 ? ' active' : '') + '" data-exp="' + i + '" style="cursor:pointer;" title="Toggle Expert #' + pad.slice(-2) + '">E' + pad.slice(-2) + '</div>');
            }
            html.push('<div class="moe-mini-cell shared">Shared Expert (Always Active)</div>');
            miniGrid.innerHTML = html.join('');
            miniGrid.querySelectorAll('.moe-mini-cell:not(.shared)').forEach(function (cell) {
                cell.addEventListener('click', function () {
                    cell.classList.toggle('active');
                    updateMoEActiveList();
                });
            });
        }

        if (routeBatchBtn && miniGrid) {
            routeBatchBtn.addEventListener('click', function () {
                var chosen = [];
                while (chosen.length < 2) {
                    var r = Math.floor(Math.random() * 8);
                    if (chosen.indexOf(r) === -1) chosen.push(r);
                }
                chosen.sort(function (a, b) { return a - b; });
                miniGrid.querySelectorAll('.moe-mini-cell:not(.shared)').forEach(function (c, idx) {
                    c.classList.toggle('active', chosen.indexOf(idx) !== -1);
                });
                updateMoEActiveList();
                var loss = (0.01 + Math.random() * 0.04).toFixed(4);
                if (dispatchValEl) {
                    dispatchValEl.textContent = 'aux_loss=' + loss + ' (\u03b1=0.01)';
                    setTimeout(function () { dispatchValEl.textContent = 'torch (default)'; }, 2500);
                }
            });
        }
    }

    document.addEventListener('DOMContentLoaded', function () {
        initAttentionSinkHero();
        initPassDiagram();
        initMechanismExploders();
        initMoeRouting();
        initSinkBiasToggle();
        initScrollspy();
        initHighlightAndMath();
    });
})();
