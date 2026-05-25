/* ═══════════════════════════════════════════════════════════════════════════
   Clasificador de Mensajes Clínicos — Frontend
   ═══════════════════════════════════════════════════════════════════════════ */

'use strict';

// ── Estado global ──────────────────────────────────────────────────────────
let mensajeIdActual = null;
let chartGlobal = null;
let chartClases = null;

// ── Helpers de UI ──────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

function labelBonito(pred) {
  if (!pred) return '—';
  return { leve: '🟢 Leve', moderada: '🟡 Moderada', grave: '🔴 Grave / Urgente' }[pred] || pred;
}

function claseBadge(pred) {
  return { leve: 'badge-leve', moderada: 'badge-moderada', grave: 'badge-grave' }[pred] || 'badge-nd';
}

function claseBadgeSm(pred) {
  return { leve: 'leve', moderada: 'moderada', grave: 'grave' }[pred] || 'nd';
}

function pct(v) {
  return v != null ? (v * 100).toFixed(1) + '%' : '—';
}

function formatFecha(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleDateString('es-ES', { day: '2-digit', month: '2-digit', year: '2-digit' })
    + ' ' + d.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' });
}

// ── Pestañas ───────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
    btn.classList.add('active');
    $('tab-' + tab).classList.remove('hidden');
    if (tab === 'estadisticas') cargarEstadisticas();
  });
});

// ── Voz (Web Speech API) ───────────────────────────────────────────────────
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let escuchando = false;
let transcripcionAcumulada = '';

if (SpeechRecognition) {
  recognition = new SpeechRecognition();
  recognition.lang = 'es-ES';
  recognition.continuous = true;      // no se corta con pausas
  recognition.interimResults = true;

  recognition.onresult = e => {
    // Acumular resultados finales + mostrar provisional en tiempo real
    let finals = '';
    let provisional = '';
    for (const result of e.results) {
      if (result.isFinal) {
        finals += result[0].transcript;
      } else {
        provisional += result[0].transcript;
      }
    }
    transcripcionAcumulada = finals;
    $('mensaje-texto').value = (transcripcionAcumulada + provisional).trim();
  };

  // onend se dispara si el navegador corta solo (timeout interno del SO)
  // → reiniciamos automáticamente mientras el usuario no haya pulsado parar
  recognition.onend = () => {
    if (escuchando) {
      // reinicio automático para mantener la sesión abierta
      try { recognition.start(); } catch (_) {}
    } else {
      detenerVoz();
    }
  };

  recognition.onerror = e => {
    if (e.error === 'no-speech') return; // pausa normal, no es un error
    escuchando = false;
    detenerVoz();
    if (e.error !== 'aborted') alert('Error de micrófono: ' + e.error);
  };
} else {
  $('btn-voz').title = 'Tu navegador no soporta dictado por voz (usa Chrome o Edge)';
  $('btn-voz').style.opacity = '0.3';
  $('btn-voz').disabled = true;
}

function detenerVoz() {
  escuchando = false;
  $('btn-voz').textContent = '🎤';
  $('btn-voz').classList.remove('activo');
  $('btn-parar-voz').classList.add('hidden');
  $('voz-estado').classList.add('hidden');
  transcripcionAcumulada = '';
}

$('btn-voz').addEventListener('click', () => {
  if (!recognition) return;
  transcripcionAcumulada = '';
  recognition.start();
  escuchando = true;
  $('btn-voz').textContent = '🎙️';
  $('btn-voz').classList.add('activo');
  $('btn-parar-voz').classList.remove('hidden');
  $('voz-estado').classList.remove('hidden');
});

$('btn-parar-voz').addEventListener('click', () => {
  if (!recognition) return;
  escuchando = false;
  recognition.stop();
  detenerVoz();
});

// ── Limpiar ────────────────────────────────────────────────────────────────
$('btn-limpiar').addEventListener('click', () => {
  if (escuchando) { escuchando = false; recognition?.stop(); detenerVoz(); }
  $('mensaje-texto').value = '';
  $('resultados').classList.add('hidden');
  $('opinion-select').value = '';
  $('notas-input').value = '';
  $('btn-guardar').disabled = true;
  $('guardado-ok').classList.add('hidden');
  mensajeIdActual = null;
});

// ── Clasificar ─────────────────────────────────────────────────────────────
$('btn-clasificar').addEventListener('click', clasificar);
$('mensaje-texto').addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'Enter') clasificar();
});

async function clasificar() {
  const texto = $('mensaje-texto').value.trim();
  if (!texto) {
    $('mensaje-texto').focus();
    return;
  }

  $('btn-clasificar').disabled = true;
  $('btn-clasificar').textContent = 'Clasificando…';

  try {
    const resp = await fetch('/api/clasificar', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texto }),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();

    mensajeIdActual = data.id;
    mostrarResultados(texto, data);
  } catch (err) {
    alert('Error al clasificar: ' + err.message);
  } finally {
    $('btn-clasificar').disabled = false;
    $('btn-clasificar').textContent = 'Clasificar';
  }
}

function mostrarResultados(texto, data) {
  $('texto-mostrado').textContent = texto;
  $('resultados').classList.remove('hidden');
  $('guardado-ok').classList.add('hidden');
  $('opinion-select').value = '';
  $('notas-input').value = '';
  $('btn-guardar').disabled = true;

  // TF-IDF
  renderModelo('tfidf', data.tfidf);

  // BETO
  const beto = data.beto;
  if (!beto.prediccion) {
    $('badge-beto').className = 'prioridad-badge badge-nd';
    $('badge-beto').textContent = 'No disponible';
    $('conf-beto').textContent = beto.error || '';
    $('prob-beto').innerHTML = '';
    $('beto-estado-chip').classList.add('hidden');
  } else {
    renderModelo('beto', beto);
    $('beto-estado-chip').classList.add('hidden');
  }

  // scroll suave al resultado
  $('resultados').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderModelo(id, datos) {
  const badge = $('badge-' + id);
  badge.className = 'prioridad-badge ' + claseBadge(datos.prediccion);
  badge.textContent = labelBonito(datos.prediccion);

  $('conf-' + id).textContent = datos.confianza != null
    ? 'Confianza: ' + pct(datos.confianza)
    : '';

  const probDiv = $('prob-' + id);
  probDiv.innerHTML = '';
  if (datos.probabilidades) {
    const orden = ['grave', 'moderada', 'leve'];
    orden.forEach(cls => {
      const p = datos.probabilidades[cls] ?? 0;
      const row = document.createElement('div');
      row.className = 'barra-row barra-' + cls;
      row.innerHTML = `
        <span class="barra-label">${labelBonito(cls).replace(/^.+ /, '')}</span>
        <span class="barra-track"><span class="barra-fill" style="width:${(p*100).toFixed(1)}%"></span></span>
        <span class="barra-pct">${pct(p)}</span>
      `;
      probDiv.appendChild(row);
    });
  }

  // Keywords (solo TF-IDF)
  const kwDiv = $('keywords-' + id);
  if (kwDiv) {
    kwDiv.innerHTML = '';
    if (datos.keywords && datos.keywords.length) {
      kwDiv.innerHTML = `
        <div class="keywords-label">Términos clave detectados:</div>
        <div class="keywords-chips">
          ${datos.keywords.map(k => `<span class="kw-chip">${k}</span>`).join('')}
        </div>
      `;
    }
  }

  // Mensaje más cercano en BD
  const cercDiv = $('cercano-' + id);
  if (cercDiv) {
    const c = datos.cercano;
    if (c) {
      const predKey = id === 'tfidf' ? 'pred_tfidf' : 'pred_beto';
      const predLabel = c[predKey] ? `<span class="badge-sm ${claseBadgeSm(c[predKey])}">${c[predKey]}</span>` : '';
      const medicoLabel = c.opinion_medico
        ? `· médico: <span class="badge-sm ${claseBadgeSm(c.opinion_medico)}">${c.opinion_medico}</span>`
        : '';
      cercDiv.innerHTML = `
        <div class="cercano-box">
          <span class="cercano-label">Mensaje más similar en BD <span class="sim-pct">${c.similitud}% similitud</span></span>
          <div class="cercano-texto">"${c.texto}"</div>
          <div class="cercano-preds">${predLabel} ${medicoLabel}</div>
        </div>
      `;
    } else {
      cercDiv.innerHTML = '<div class="cercano-vacio">Sin mensajes similares en la base de datos aún</div>';
    }
  }
}

// ── Guardar valoración del médico ──────────────────────────────────────────
$('opinion-select').addEventListener('change', () => {
  $('btn-guardar').disabled = !$('opinion-select').value;
});

$('btn-guardar').addEventListener('click', async () => {
  if (!mensajeIdActual) return;
  const opinion = $('opinion-select').value;
  const notas = $('notas-input').value.trim() || null;

  $('btn-guardar').disabled = true;
  $('btn-guardar').textContent = 'Guardando…';

  try {
    const resp = await fetch(`/api/mensajes/${mensajeIdActual}/opinion`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ opinion_medico: opinion, notas }),
    });
    if (!resp.ok) throw new Error(await resp.text());

    $('guardado-ok').classList.remove('hidden');
    $('btn-guardar').textContent = 'Guardar valoración';
  } catch (err) {
    alert('Error al guardar: ' + err.message);
    $('btn-guardar').disabled = false;
    $('btn-guardar').textContent = 'Guardar valoración';
  }
});

// ── Estadísticas ───────────────────────────────────────────────────────────
$('btn-refrescar').addEventListener('click', cargarEstadisticas);


async function cargarEstadisticas() {
  try {
    const [statsResp, mensajesResp] = await Promise.all([
      fetch('/api/estadisticas'),
      fetch('/api/mensajes'),
    ]);
    const stats = await statsResp.json();
    const mensajes = await mensajesResp.json();

    actualizarResumen(stats);
    actualizarGraficas(stats);
    actualizarTabla(mensajes);
  } catch (err) {
    console.error('Error cargando estadísticas:', err);
  }
}

function actualizarResumen(stats) {
  $('stat-clasificados').textContent = stats.total_clasificados;
  $('stat-valorados').textContent = stats.total_valorados;
  $('stat-acc-tfidf').textContent = stats.tfidf_accuracy != null
    ? stats.tfidf_accuracy + '%' : '—';
  $('stat-acc-beto').textContent = stats.beto_accuracy != null
    ? stats.beto_accuracy + '%' : '—';
}

function actualizarGraficas(stats) {
  // Gráfica 1 — precisión global
  const datosGlobal = {
    labels: ['TF-IDF', 'BETO'],
    datasets: [{
      label: 'Precisión (%)',
      data: [stats.tfidf_accuracy ?? 0, stats.beto_accuracy ?? 0],
      backgroundColor: ['rgba(37,99,235,.7)', 'rgba(124,58,237,.7)'],
      borderColor: ['#2563eb', '#7c3aed'],
      borderWidth: 2,
      borderRadius: 6,
    }],
  };

  if (chartGlobal) chartGlobal.destroy();
  chartGlobal = new Chart($('chart-global'), {
    type: 'bar',
    data: datosGlobal,
    options: {
      responsive: true,
      scales: { y: { min: 0, max: 100, ticks: { callback: v => v + '%' } } },
      plugins: { legend: { display: false } },
    },
  });

  // Gráfica 2 — por clase
  const clases = ['leve', 'moderada', 'grave'];
  const labels = ['🟢 Leve', '🟡 Moderada', '🔴 Grave'];
  const tfidfData = clases.map(c => stats.por_clase[c]?.tfidf_ok ?? 0);
  const betoData  = clases.map(c => stats.por_clase[c]?.beto_ok  ?? 0);

  if (chartClases) chartClases.destroy();
  chartClases = new Chart($('chart-clases'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'TF-IDF',
          data: tfidfData,
          backgroundColor: 'rgba(37,99,235,.7)',
          borderColor: '#2563eb',
          borderWidth: 2,
          borderRadius: 4,
        },
        {
          label: 'BETO',
          data: betoData,
          backgroundColor: 'rgba(124,58,237,.7)',
          borderColor: '#7c3aed',
          borderWidth: 2,
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      scales: { y: { min: 0, max: 100, ticks: { callback: v => v + '%' } } },
      plugins: { legend: { position: 'bottom' } },
    },
  });
}

let _mensajesCache = [];

function actualizarTabla(mensajes) {
  _mensajesCache = mensajes;
  const tbody = $('tabla-body');
  if (!mensajes.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="tabla-vacia">Sin datos aún</td></tr>';
    return;
  }

  tbody.innerHTML = mensajes.map((m, idx) => {
    const tOk = m.tfidf_correcto === true ? '<span class="check-ok">✓</span>'
              : m.tfidf_correcto === false ? '<span class="check-err">✗</span>'
              : '<span class="check-nd">—</span>';
    const bOk = m.beto_correcto === true ? '<span class="check-ok">✓</span>'
              : m.beto_correcto === false ? '<span class="check-err">✗</span>'
              : '<span class="check-nd">—</span>';
    const medico = m.opinion_medico
      ? `<span class="badge-sm ${claseBadgeSm(m.opinion_medico)}">${m.opinion_medico}</span>`
      : '<span class="check-nd">—</span>';
    return `
      <tr data-id="${m.id}">
        <td onclick="abrirDetalle(${m.id})">${mensajes.length - idx}</td>
        <td onclick="abrirDetalle(${m.id})">
          <div class="msg-truncada">${m.texto}</div>
          ${m.notas ? `<div class="notas-inline">💬 ${m.notas}</div>` : ''}
        </td>
        <td onclick="abrirDetalle(${m.id})"><span class="badge-sm ${claseBadgeSm(m.pred_tfidf)}">${m.pred_tfidf} ${pct(m.conf_tfidf)}</span></td>
        <td onclick="abrirDetalle(${m.id})">${m.pred_beto ? `<span class="badge-sm ${claseBadgeSm(m.pred_beto)}">${m.pred_beto} ${pct(m.conf_beto)}</span>` : '<span class="check-nd">—</span>'}</td>
        <td onclick="abrirDetalle(${m.id})">${medico}</td>
        <td onclick="abrirDetalle(${m.id})">${tOk}</td>
        <td onclick="abrirDetalle(${m.id})">${bOk}</td>
        <td onclick="abrirDetalle(${m.id})">${formatFecha(m.creado_en)}</td>
        <td class="acciones-cel">
          <button class="btn-editar" onclick="editarMensaje(${m.id})" title="Editar valoración">✏️</button>
          <button class="btn-editar btn-del" onclick="borrarMensaje(${m.id})" title="Eliminar mensaje">🗑</button>
        </td>
      </tr>
    `;
  }).join('');
}

// ── Modal detalle / edición ────────────────────────────────────────────────
let _modalIdActual = null;

function _rellenarModal(m) {
  _modalIdActual = m.id;
  $('modal-texto').textContent = m.texto;
  $('modal-tfidf').innerHTML = `<span class="badge-sm ${claseBadgeSm(m.pred_tfidf)}">${m.pred_tfidf}</span> ${pct(m.conf_tfidf)}`;
  $('modal-beto').innerHTML = m.pred_beto
    ? `<span class="badge-sm ${claseBadgeSm(m.pred_beto)}">${m.pred_beto}</span> ${pct(m.conf_beto)}`
    : '<span class="check-nd">—</span>';
  $('modal-medico').innerHTML = m.opinion_medico
    ? `<span class="badge-sm ${claseBadgeSm(m.opinion_medico)}">${m.opinion_medico}</span>`
    : '<span class="check-nd">Sin valorar</span>';
  $('modal-fecha').textContent = formatFecha(m.creado_en);
  if (m.notas) {
    $('modal-notas').textContent = m.notas;
    $('modal-notas-wrap').classList.remove('hidden');
  } else {
    $('modal-notas-wrap').classList.add('hidden');
  }
  $('modal-guardado-ok').classList.add('hidden');
}

window.abrirDetalle = function(id) {
  const m = _mensajesCache.find(x => x.id === id);
  if (!m) return;
  _rellenarModal(m);

  // Modo lectura
  $('modal-titulo').textContent = 'Detalle del mensaje';
  $('modal-vista-medico').classList.remove('hidden');
  $('modal-notas-wrap').classList.toggle('hidden', !m.notas);
  $('modal-edit-fields').classList.add('hidden');
  $('modal-guardar').classList.add('hidden');

  $('modal-detalle').classList.remove('hidden');
};

window.editarMensaje = function(id) {
  const m = _mensajesCache.find(x => x.id === id);
  if (!m) return;
  _rellenarModal(m);

  // Modo edición
  $('modal-titulo').textContent = 'Editar valoración';
  $('modal-vista-medico').classList.add('hidden');
  $('modal-notas-wrap').classList.add('hidden');
  $('modal-edit-fields').classList.remove('hidden');
  $('modal-edit-fields').style.display = 'flex';
  $('modal-edit-opinion').value = m.opinion_medico || '';
  $('modal-edit-notas').value = m.notas || '';
  $('modal-guardar').classList.remove('hidden');

  $('modal-detalle').classList.remove('hidden');
};

$('modal-guardar').addEventListener('click', async () => {
  if (!_modalIdActual) return;
  const opinion = $('modal-edit-opinion').value;
  const notas = $('modal-edit-notas').value.trim() || null;

  $('modal-guardar').disabled = true;
  $('modal-guardar').textContent = 'Guardando…';

  try {
    const resp = await fetch(`/api/mensajes/${_modalIdActual}/opinion`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ opinion_medico: opinion || null, notas }),
    });
    if (!resp.ok) throw new Error(await resp.text());

    $('modal-guardado-ok').classList.remove('hidden');
    // Refrescar tabla sin cerrar modal
    await cargarEstadisticas();
    // Actualizar cache para que el modal muestre datos frescos
    const m = _mensajesCache.find(x => x.id === _modalIdActual);
    if (m) { m.opinion_medico = opinion || null; m.notas = notas; }
  } catch (err) {
    alert('Error al guardar: ' + err.message);
  } finally {
    $('modal-guardar').disabled = false;
    $('modal-guardar').textContent = 'Guardar cambios';
  }
});

window.borrarMensaje = async function(id) {
  if (!confirm(`¿Eliminar el mensaje #${id}? Esta acción no se puede deshacer.`)) return;
  try {
    const resp = await fetch(`/api/mensajes/${id}`, { method: 'DELETE' });
    if (!resp.ok) throw new Error(await resp.text());
    await cargarEstadisticas();
  } catch (err) {
    alert('Error al eliminar: ' + err.message);
  }
};

$('modal-cerrar').addEventListener('click', () => $('modal-detalle').classList.add('hidden'));
$('modal-detalle').addEventListener('click', e => {
  if (e.target === $('modal-detalle')) $('modal-detalle').classList.add('hidden');
});
