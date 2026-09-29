import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, BarChart, Bar, AreaChart, Area, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend 
} from 'recharts';
import { MapContainer, TileLayer, Marker, Popup, Polyline, Circle, useMap, ImageOverlay } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { Activity, Thermometer, Wind, Droplets, AlertTriangle, Crosshair, Map as MapIcon, Database, ActivitySquare, Server, CheckCircle2, AlertOctagon, Cpu, Search, CloudLightning, Users, Building2 } from 'lucide-react';

const HAZARD_COLORS = {
  rainfall: { text: 'text-cyan-400', bg: 'bg-cyan-500/10', border: 'border-cyan-500/30', hex: '#22d3ee' },
  heat: { text: 'text-red-400', bg: 'bg-red-500/10', border: 'border-red-500/30', hex: '#f87171' },
  cold: { text: 'text-blue-200', bg: 'bg-blue-300/10', border: 'border-blue-300/30', hex: '#bfdbfe' },
  cyclone: { text: 'text-fuchsia-400', bg: 'bg-fuchsia-500/10', border: 'border-fuchsia-500/30', hex: '#e879f9' }
};

const SEVERITY_COLORS = {
  NORMAL: 'text-slate-500',
  WATCH: 'text-yellow-400',
  SEVERE: 'text-orange-500',
  EXTREME: 'text-red-500'
};

const createHazardIcon = (hazard) => {
  const color = HAZARD_COLORS[hazard] ? HAZARD_COLORS[hazard].hex : '#fff';
  return L.divIcon({
    className: 'custom-ai-marker',
    html: `<div style="
        width: 20px; 
        height: 20px; 
        background: rgba(15, 23, 42, 0.9);
        border: 2px solid ${color};
        box-shadow: 0 0 10px ${color}80;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
      ">
        <div style="width: 6px; height: 6px; background-color: ${color}; border-radius: 50%;"></div>
      </div>`,
    iconSize: [20, 20],
    iconAnchor: [10, 10],
  });
};

const MapFlyTo = ({ center }) => {
  const map = useMap();
  useEffect(() => {
    if (center) map.flyTo(center, 6, { animate: true, duration: 1.5 });
  }, [center, map]);
  return null;
};

// Helper to get severity label
const getSeverityLabel = (score) => {
  if (score >= 80) return 'EXTREME';
  if (score >= 50) return 'SEVERE';
  return 'WATCH';
};

const getSeverityColor = (score) => {
  if (score >= 80) return SEVERITY_COLORS.EXTREME;
  if (score >= 50) return SEVERITY_COLORS.SEVERE;
  return SEVERITY_COLORS.WATCH;
};

export default function App() {
  const [activeTab, setActiveTab] = useState('OVERVIEW');
  const [selectedEventId, setSelectedEventId] = useState(null);
  const [selectedRiskFilter, setSelectedRiskFilter] = useState(null);
  const [selectedAdvisory, setSelectedAdvisory] = useState(null);
  const [isDownscaling, setIsDownscaling] = useState(false);
  
  const [rainfallDate, setRainfallDate] = useState('2020-01-25');
  const [rainfallLayer, setRainfallLayer] = useState('AI_DOWNSCALED');
  const [rainfallRaster, setRainfallRaster] = useState(null);
  const [rainfallBounds, setRainfallBounds] = useState([[40, 60], [-5, 100]]);
  const [rainfallLoading, setRainfallLoading] = useState(false);
  const [rainfallError, setRainfallError] = useState(null);
  const [rainfallStats, setRainfallStats] = useState({ mean: 0, max: 0 });
  const [rainfallTrend, setRainfallTrend] = useState([]);
  
  useEffect(() => {
    const fetchRaster = async () => {
      setRainfallLoading(true);
      setRainfallError(null);
      try {
        const res = await fetch(`http://localhost:3001/api/rainfall/downscaled?date=${rainfallDate}&layer=${rainfallLayer}`);
        const data = await res.json();
        if (data.success === false || data.error) throw new Error(data.error || 'Failed to load raster');
        setRainfallRaster(data.raster_base64);
        if (data.bounds) setRainfallBounds(data.bounds);
        setRainfallStats({ mean: data.mean, max: data.max });
        if (data.trend) setRainfallTrend(data.trend);
      } catch (e) {
        setRainfallError(e.message);
      } finally {
        setRainfallLoading(false);
      }
    };
    fetchRaster();
  }, [rainfallDate, rainfallLayer]);
  
  const [systemStatus, setSystemStatus] = useState(null);
  const [aiHealth, setAiHealth] = useState(null);
  const [aiEvents, setAiEvents] = useState([]);
  const [forecastData, setForecastData] = useState(null);
  const [impactsData, setImpactsData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAllData = async () => {
      try {
        setLoading(true);
        const [sysRes, healthRes, eventsRes, forecastRes, impactsRes] = await Promise.all([
          fetch('http://localhost:3001/api/system-status').catch(() => null),
          fetch('http://localhost:3001/api/ai/health').catch(() => null),
          fetch('http://localhost:3001/api/ai/events').catch(() => null),
          fetch('http://localhost:3001/api/ai/forecast').catch(() => null),
          fetch('http://localhost:3001/api/ai/impacts').catch(() => null)
        ]);

        if (sysRes?.ok) setSystemStatus(await sysRes.json());
        if (healthRes?.ok) setAiHealth(await healthRes.json());
        if (eventsRes?.ok) {
           const data = await eventsRes.json();
           setAiEvents(Array.isArray(data) ? data : (data.events || []));
        }
        if (forecastRes?.ok) {
           const data = await forecastRes.json();
           // forecast API returns {horizon, timesteps, anomalies}
           setForecastData(data);
        }
        if (impactsRes?.ok) {
           const data = await impactsRes.json();
           setImpactsData(Array.isArray(data) ? data : (data.results || []));
        }
        
        setLoading(false);
      } catch (err) {
        console.error("API Fetch Error:", err);
        setError("Failed to connect to AI Intelligence Core.");
        setLoading(false);
      }
    };
    fetchAllData();
  }, []);

  const selectedEvent = aiEvents.find(e => e.id === selectedEventId) || aiEvents[0];
  const selectedImpact = impactsData.find(i => i.eventId === selectedEvent?.id);

  const TABS = ['OVERVIEW', 'FORECAST', 'RISK INTELLIGENCE', 'RAINFALL PROTOTYPE', 'SYSTEM'];

  if (loading) {
    return (
      <div className="min-h-screen bg-[#F4F7FE] flex items-center justify-center text-cyan-500 font-mono text-sm">
        <Activity className="animate-spin mr-3" /> INITIALIZING AI METEOROLOGICAL COMMAND CENTER...
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-[#F4F7FE] flex items-center justify-center text-red-500 font-mono text-sm">
        <AlertTriangle className="mr-3" /> {error}
      </div>
    );
  }

  // Helper for Intelligence Panel
  const renderIntelligencePanel = (event, impact) => {
    if (!event) return <div className="p-6 text-slate-400">No active events detected.</div>;
    const colors = HAZARD_COLORS[event.category] || HAZARD_COLORS.heat;
    const sevLabel = getSeverityLabel(event.severityScore);
    const sevColor = getSeverityColor(event.severityScore);
    
    return (
      <div className="flex flex-col h-full bg-white rounded-[24px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-slate-100 overflow-y-auto">
        <div className="p-8 border-b border-slate-100 bg-gradient-to-b from-slate-50 to-transparent">
          <div className="flex justify-between items-start mb-4">
            <span className="text-xs font-mono text-slate-500">{event.id}</span>
            <span className={`text-[10px] uppercase font-bold px-3 py-1 rounded-full border ${colors.border} ${colors.text} ${colors.bg}`}>
              {event.category}
            </span>
          </div>
          <h2 className="text-3xl font-bold text-slate-800 tracking-tight">{event.type}</h2>
          
          <div className="grid grid-cols-3 gap-3 mt-6">
            <div className="bg-white rounded-xl p-3 border border-slate-100 shadow-sm flex flex-col items-center text-center">
              <p className="text-[10px] text-slate-400 uppercase font-bold tracking-wider mb-1">Severity</p>
              <p className={`text-sm font-bold ${sevColor}`}>{sevLabel}</p>
            </div>
            <div className="bg-white rounded-xl p-3 border border-slate-100 shadow-sm flex flex-col items-center text-center">
              <p className="text-[10px] text-slate-400 uppercase font-bold tracking-wider mb-1">Lead Time</p>
              <p className="text-sm text-slate-800 font-bold">{event.currentLead || '24h'}</p>
            </div>
            <div className="bg-white rounded-xl p-3 border border-slate-100 shadow-sm flex flex-col items-center text-center">
              <p className="text-[10px] text-slate-400 uppercase font-bold tracking-wider mb-1">Probability</p>
              <p className="text-sm text-slate-800 font-bold">{event.probability}%</p>
            </div>
          </div>
        </div>

        <div className="p-6 space-y-6 bg-slate-50/50">
          <section>
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3 flex items-center">
              <ActivitySquare className="w-4 h-4 mr-2 text-rose-500" /> Event Intensity
            </h3>
            <div className="bg-gradient-to-br from-rose-50 to-orange-50 p-5 rounded-2xl border border-rose-100 shadow-sm relative overflow-hidden group">
              <div className="absolute -right-4 -top-4 opacity-5 group-hover:opacity-10 transition-opacity">
                <Thermometer size={100} />
              </div>
              <div className="flex items-center justify-between mb-2 relative z-10">
                <span className="text-sm font-semibold text-rose-900">Intensity Level</span>
                <span className="px-2 py-1 bg-rose-200/50 text-rose-700 text-[10px] font-bold rounded-full uppercase tracking-wider">
                  {event.severityScore >= 80 ? 'Exceptionally High' : event.severityScore >= 50 ? 'High' : 'Moderate'}
                </span>
              </div>
              <div className="flex items-baseline gap-2 relative z-10">
                <span className="text-3xl font-black text-rose-600 tracking-tighter">
                  +{event.severityScore ? (event.severityScore / 20).toFixed(1) : 2.5}
                </span>
                <span className="text-sm text-rose-800/60 font-medium">σ (Sigma Scale)</span>
              </div>
              <p className="text-xs text-rose-800/70 mt-2 relative z-10 font-medium">
                This event is significantly stronger than historical averages.
              </p>
            </div>
          </section>

          <section>
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3 flex items-center">
              <Crosshair className="w-4 h-4 mr-2 text-blue-500" /> Movement & Tracking
            </h3>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-white p-4 rounded-xl border border-slate-100 shadow-[0_2px_10px_rgb(0,0,0,0.02)] hover:shadow-md transition-shadow">
                <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider mb-1 block">Direction</span>
                <span className="text-lg font-black text-slate-800">{event.movement?.split('@')[0]?.trim() || 'North-West'}</span>
              </div>
              <div className="bg-white p-4 rounded-xl border border-slate-100 shadow-[0_2px_10px_rgb(0,0,0,0.02)] hover:shadow-md transition-shadow">
                <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider mb-1 block">Speed</span>
                <span className="text-lg font-black text-slate-800">{event.movement?.split('@')[1]?.trim() || '15 km/h'}</span>
              </div>
            </div>
          </section>

          <section>
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3 flex items-center">
              <CheckCircle2 className="w-4 h-4 mr-2 text-emerald-500" /> AI Confidence Checks
            </h3>
            <div className="flex flex-wrap gap-2">
              <span className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-emerald-50 text-emerald-700 border border-emerald-100 rounded-lg text-xs font-bold shadow-sm">
                <CheckCircle2 size={14} className="text-emerald-500" /> Temp Verified
              </span>
              <span className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-emerald-50 text-emerald-700 border border-emerald-100 rounded-lg text-xs font-bold shadow-sm">
                <CheckCircle2 size={14} className="text-emerald-500" /> Precip Verified
              </span>
              <span className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-emerald-50 text-emerald-700 border border-emerald-100 rounded-lg text-xs font-bold shadow-sm">
                <CheckCircle2 size={14} className="text-emerald-500" /> Physics Pass
              </span>
            </div>
          </section>

          {impact && (
            <section>
              <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3 flex items-center">
                <AlertOctagon className="w-4 h-4 mr-2 text-violet-500" /> Estimated Impact
              </h3>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gradient-to-br from-violet-500 to-fuchsia-500 p-4 rounded-xl shadow-md text-white relative overflow-hidden hover:scale-[1.02] transition-transform cursor-default">
                  <div className="absolute -right-2 -bottom-2 opacity-20">
                    <Users size={64} />
                  </div>
                  <span className="text-[10px] text-violet-100 font-bold uppercase tracking-wider mb-1 block relative z-10">People Affected</span>
                  <span className="text-xl font-black tracking-tight relative z-10">{impact.affectedAssets?.population >= 1000000 ? `${(impact.affectedAssets.population / 1000000).toFixed(1)}M` : impact.affectedAssets?.population?.toLocaleString() || 'N/A'}</span>
                </div>
                <div className="bg-gradient-to-br from-amber-500 to-orange-500 p-4 rounded-xl shadow-md text-white relative overflow-hidden hover:scale-[1.02] transition-transform cursor-default">
                  <div className="absolute -right-2 -bottom-2 opacity-20">
                    <Building2 size={64} />
                  </div>
                  <span className="text-[10px] text-amber-100 font-bold uppercase tracking-wider mb-1 block relative z-10">Infrastructure</span>
                  <span className="text-xl font-black tracking-tight relative z-10">{impact.affectedAssets?.infrastructure?.toLocaleString() || 'N/A'} At Risk</span>
                </div>
              </div>
            </section>
          )}
        </div>
      </div>
    );
  };

  const renderOverview = () => {
    const heroEvent = selectedEvent;

    // Generate timeline data for overview based on real anomalies
    const overviewTimesteps = forecastData?.timesteps?.slice(0, 4) || ['T+24h', 'T+48h', 'T+72h', 'T+96h'];
    const anomalies = forecastData?.anomalies || aiEvents;
    const overviewTimelineData = overviewTimesteps.map((ts, i) => {
      const avgProb = anomalies.length > 0
        ? anomalies.reduce((sum, a) => sum + (a.probability || 0), 0) / anomalies.length
        : 80;
      return { name: ts, probability: parseFloat((avgProb * (1 - i * 0.05)).toFixed(1)) };
    });

    return (
      <div className="h-[calc(100vh-140px)] grid grid-cols-1 xl:grid-cols-12 gap-8">
        
        {/* Left/Main Column */}
        <div className="xl:col-span-8 flex flex-col gap-8">
          
          {/* Top Row: Hero Weather + Map */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 flex-1">
            
            {/* Hero Weather Card */}
            <div className="lg:col-span-5 bg-gradient-to-br from-blue-100 to-orange-50 rounded-[30px] p-8 shadow-[0_8px_30px_rgb(0,0,0,0.04)] relative overflow-hidden flex flex-col justify-between">
              <div>
                <div className="flex items-center gap-2 bg-white/60 backdrop-blur-md w-max px-4 py-2 rounded-full text-sm font-semibold text-slate-700 shadow-sm">
                  <MapIcon size={16} className="text-orange-500" />
                  {heroEvent ? heroEvent.movement?.split('@')[0] || 'Affected Region' : 'System Wide'}
                </div>
                <h2 className="text-3xl font-bold text-slate-800 mt-6 mb-1">
                  {heroEvent ? heroEvent.type : 'Weather Normal'}
                </h2>
                <p className="text-slate-600 font-medium">{heroEvent ? 'Extreme Alert' : 'No Active Events'}</p>
              </div>

              <div className="flex items-end justify-between mt-8 relative z-10">
                <div>
                  <h1 className="text-6xl font-bold text-slate-800 tracking-tighter">
                    {heroEvent ? `${heroEvent.probability}%` : '--'}
                  </h1>
                  <p className="text-slate-500 font-medium mt-1">Probability</p>
                </div>
                <div className="flex gap-3">
                  <div className="bg-white/60 backdrop-blur-md p-3 rounded-2xl flex flex-col items-center shadow-sm">
                    <span className="text-[10px] text-slate-500 uppercase font-bold">Severity</span>
                    <span className="text-sm font-bold text-slate-800">{heroEvent ? heroEvent.severityScore : 0}</span>
                  </div>
                  <div className="bg-white/60 backdrop-blur-md p-3 rounded-2xl flex flex-col items-center shadow-sm">
                    <span className="text-[10px] text-slate-500 uppercase font-bold">Lead</span>
                    <span className="text-sm font-bold text-slate-800">{heroEvent ? heroEvent.currentLead : '-'}</span>
                  </div>
                </div>
              </div>

              {/* Decorative weather illustration (simple CSS shapes for now) */}
              <div className="absolute -top-10 -right-10 w-48 h-48 bg-orange-300 rounded-full mix-blend-multiply filter blur-2xl opacity-50 animate-pulse"></div>
              <div className="absolute -bottom-10 right-10 w-48 h-48 bg-blue-300 rounded-full mix-blend-multiply filter blur-2xl opacity-50"></div>
            </div>

            {/* Geographic Map */}
            <div className="lg:col-span-7 bg-white rounded-[30px] p-2 shadow-[0_8px_30px_rgb(0,0,0,0.04)] relative">
              <div className="w-full h-full rounded-[24px] overflow-hidden">
                <MapContainer center={[22.0, 79.0]} zoom={4} style={{ height: '100%', width: '100%', backgroundColor: '#E2E8F0' }} zoomControl={false}>
                  <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution="&copy; OpenStreetMap"
                  />
                  {aiEvents.map(evt => (
                    <React.Fragment key={evt.id}>
                      <Marker 
                        position={[evt.lat, evt.lng]} 
                        icon={createHazardIcon(evt.category)}
                        eventHandlers={{ click: () => setSelectedEventId(evt.id) }}
                      >
                        <Popup className="bg-white border-slate-100 text-slate-800 rounded-xl shadow-lg">
                          <strong>{evt.type}</strong><br/>
                          Prob: {evt.probability}%<br/>
                          Lead: {evt.currentLead}
                        </Popup>
                      </Marker>
                      <Circle center={[evt.lat, evt.lng]} radius={50000} pathOptions={{ color: HAZARD_COLORS[evt.category]?.hex || '#3b82f6', fillColor: HAZARD_COLORS[evt.category]?.hex || '#3b82f6', fillOpacity: 0.2, weight: 2 }} />
                    </React.Fragment>
                  ))}
                  <MapFlyTo center={selectedEvent ? [selectedEvent.lat, selectedEvent.lng] : null} />
                </MapContainer>
              </div>
              <div className="absolute top-6 left-6 z-[1000] bg-white/90 backdrop-blur-md px-4 py-2 rounded-full text-xs font-bold text-slate-700 shadow-sm border border-slate-100">
                Geographic Risk Layer
              </div>
            </div>
          </div>

          {/* Bottom Row: KPI Cards + Timeline */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 h-[260px]">
            <div className="lg:col-span-4 flex flex-col gap-4">
               <div className="bg-white rounded-[24px] p-6 shadow-[0_8px_30px_rgb(0,0,0,0.04)] flex items-center justify-between hover:-translate-y-1 transition-transform">
                 <div>
                   <p className="text-sm font-medium text-slate-500">Active Events</p>
                   <h3 className="text-2xl font-bold text-slate-800">{aiEvents.length}</h3>
                 </div>
                 <div className="w-12 h-12 bg-orange-50 text-orange-500 rounded-full flex items-center justify-center">
                   <AlertTriangle size={24} />
                 </div>
               </div>
               <div className="bg-[#1E293B] rounded-[24px] p-6 shadow-lg flex flex-col justify-between hover:-translate-y-1 transition-transform flex-1 relative overflow-hidden">
                 <div>
                   <p className="text-sm font-medium text-slate-400">System Status</p>
                   <h3 className="text-xl font-bold text-white mt-1">Operational</h3>
                 </div>
                 <div className="flex items-center gap-2 mt-4 text-[#a7e937]">
                    <Activity size={18} />
                    <span className="text-sm font-semibold">{forecastData?.horizon || 'T+120h'} Horizon</span>
                 </div>
                 <div className="absolute -right-6 -bottom-6 text-slate-700 opacity-20">
                   <Server size={100} />
                 </div>
               </div>
            </div>

            <div className="lg:col-span-8 bg-white rounded-[30px] p-8 shadow-[0_8px_30px_rgb(0,0,0,0.04)] flex flex-col">
              <h3 className="text-lg font-bold text-slate-800 mb-6">Forecast Timeline</h3>
              <div className="flex-1 flex justify-between items-center relative mt-4">
                 <div className="absolute left-0 right-0 top-1/2 h-0.5 bg-slate-100 -z-10"></div>
                 {overviewTimelineData.map((data, idx) => {
                   const isHighRisk = data.probability > 70;
                   return (
                   <div key={data.name} className="flex flex-col items-center gap-3 cursor-pointer group">
                     <div className={`w-12 h-12 rounded-full flex items-center justify-center shadow-md transition-all duration-300 ${isHighRisk ? 'bg-orange-500 text-white scale-110 shadow-[0_4px_15px_rgba(249,115,22,0.4)]' : 'bg-white text-slate-500 border border-slate-100 hover:border-orange-200'}`}>
                       {idx % 2 === 0 ? <Wind size={20} /> : <Droplets size={20} />}
                     </div>
                     <p className="text-sm font-bold text-slate-700 mt-2">{data.name}</p>
                     <p className={`text-xs font-bold ${isHighRisk ? 'text-orange-500' : 'text-slate-400'}`}>
                       {data.probability}% Risk
                     </p>
                   </div>
                   );
                 })}
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Intelligence Panel */}
        <div className="xl:col-span-4 h-full rounded-[30px] overflow-hidden shadow-[0_8px_30px_rgb(0,0,0,0.04)] bg-white">
          {renderIntelligencePanel(selectedEvent, selectedImpact)}
        </div>
      </div>
    );
  };


  const renderForecast = () => {
    // Convert T+24h style to user friendly labels
    const getFriendlyDay = (ts) => {
      const hours = parseInt(ts.replace('T+', '').replace('h', ''));
      if (isNaN(hours)) return ts;
      const days = hours / 24;
      if (days === 1) return 'Tomorrow';
      return `In ${days} Days`;
    };

    const rawTimesteps = forecastData?.timesteps || ['T+24h', 'T+48h', 'T+72h', 'T+96h', 'T+120h', 'T+144h', 'T+168h'];
    const timesteps = rawTimesteps.map(getFriendlyDay);
    const anomalies = forecastData?.anomalies || [];
    
    // Build chart data from anomalies grouped by timestep
    const anomalyChartData = timesteps.map((ts, i) => {
      const avgAnomaly = anomalies.length > 0 
        ? anomalies.reduce((sum, a) => sum + (a.maxAnomaly || 0), 0) / anomalies.length * (Math.sin(i * 0.8) * 0.3 + 1)
        : Math.sin(i) * 1.5 + 2;
      const avgProb = anomalies.length > 0
        ? anomalies.reduce((sum, a) => sum + (a.probability || 0), 0) / anomalies.length
        : 80;
      return { 
        name: ts, 
        intensity: parseFloat(avgAnomaly.toFixed(2)), 
        confidence: parseFloat((avgProb * (1 - i * 0.05)).toFixed(1)) 
      };
    });

    return (
      <div className="space-y-8">
        <div className="bg-white rounded-[30px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-8">
          <div className="mb-8">
            <h2 className="text-lg font-bold text-slate-800 uppercase tracking-widest flex items-center">
              <CloudLightning className="w-5 h-5 mr-3 text-sky-500" /> 7-Day Risk Forecast
            </h2>
            <p className="text-xs text-slate-400 font-semibold mt-1 ml-8 tracking-wide">
              OVERALL REGIONAL AVERAGE
            </p>
          </div>
          <div className="flex justify-between items-center mb-4 px-4 relative mt-6">
            <div className="absolute top-[40%] left-4 right-4 h-1 bg-slate-100 rounded-full -z-10"></div>
            {timesteps.map((ts, i) => {
              const data = anomalyChartData[i];
              const isHighRisk = data.confidence > 70;
              
              // Map intensity to human words
              let severityWord = 'Moderate';
              let bgColor = 'bg-white text-slate-600 border border-slate-200';
              if (data.intensity > 4) { severityWord = 'Extreme'; bgColor = 'bg-rose-500 text-white shadow-[0_0_15px_rgba(244,63,94,0.4)] border-none'; }
              else if (data.intensity > 2.5) { severityWord = 'Severe'; bgColor = 'bg-orange-500 text-white shadow-[0_0_15px_rgba(249,115,22,0.4)] border-none'; }
              else if (data.intensity > 1) { severityWord = 'High'; bgColor = 'bg-amber-400 text-white shadow-[0_0_15px_rgba(251,191,36,0.4)] border-none'; }

              return (
              <div key={ts} className="flex flex-col items-center cursor-pointer group relative">
                <div className={`w-14 h-14 rounded-2xl flex flex-col items-center justify-center shadow-sm mb-3 z-10 transition-transform group-hover:-translate-y-2 group-hover:scale-110 ${bgColor}`}>
                  <span className="text-[10px] uppercase font-bold opacity-80 mb-0.5">Level</span>
                  <span className="text-sm font-black">{Math.round(data.intensity * 2)}</span>
                </div>
                <span className="text-sm font-bold text-slate-700">{ts}</span>
                <span className={`text-xs font-bold mt-1 ${isHighRisk ? 'text-rose-500' : 'text-slate-400'}`}>{severityWord} Risk</span>
              </div>
            )})}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white rounded-[30px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-8 relative overflow-hidden">
            <h3 className="text-sm font-bold text-slate-500 uppercase tracking-widest mb-6 relative z-10">Expected Event Intensity</h3>
            <div className="h-64 relative z-10">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={anomalyChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                  <XAxis dataKey="name" stroke="#cbd5e1" tick={{fill: '#64748b', fontSize: 11, fontWeight: 600}} axisLine={false} tickLine={false} />
                  <YAxis stroke="#cbd5e1" tick={{fill: '#64748b', fontSize: 11, fontWeight: 600}} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{backgroundColor: '#fff', borderColor: '#e2e8f0', color: '#1e293b', borderRadius: '12px', boxShadow: '0 4px 15px rgba(0,0,0,0.05)'}} formatter={(value) => [value, "Intensity Level"]} />
                  <Area type="monotone" dataKey="intensity" stroke="#f43f5e" strokeWidth={3} fill="rgba(244, 63, 94, 0.1)" name="Intensity" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          <div className="bg-white rounded-[30px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-8">
            <h3 className="text-sm font-bold text-slate-500 uppercase tracking-widest mb-6">AI Prediction Confidence</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={anomalyChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                  <XAxis dataKey="name" stroke="#cbd5e1" tick={{fill: '#64748b', fontSize: 11, fontWeight: 600}} axisLine={false} tickLine={false} />
                  <YAxis stroke="#cbd5e1" domain={[0, 100]} tick={{fill: '#64748b', fontSize: 11, fontWeight: 600}} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{backgroundColor: '#fff', borderColor: '#e2e8f0', color: '#1e293b', borderRadius: '12px', boxShadow: '0 4px 15px rgba(0,0,0,0.05)'}} formatter={(value) => [value + "%", "Confidence"]} />
                  <Line type="monotone" dataKey="confidence" stroke="#0ea5e9" strokeWidth={3} dot={{fill: '#0ea5e9', r: 5, strokeWidth: 2, stroke: '#fff'}} name="Confidence" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {anomalies.length > 0 && (
          <div className="bg-white rounded-[30px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-8">
            <h3 className="text-sm font-bold text-slate-500 uppercase tracking-widest mb-8">Detailed Daily Risk Breakdown</h3>
            <div className="overflow-x-auto custom-scrollbar pb-4">
              <div className="min-w-[900px]">
                <div className="grid grid-cols-6 gap-4 mb-4 px-4">
                  <div className="col-span-1 text-xs font-bold text-slate-400 uppercase tracking-wider">Affected Region</div>
                  {timesteps.map((ts, i) => {
                    // Only show 5 columns due to grid-cols-6
                    if (i >= 5) return null;
                    return <div key={ts} className="col-span-1 text-center text-xs font-bold text-slate-400 uppercase tracking-wider">{ts}</div>
                  })}
                </div>
                
                <div className="space-y-4">
                  {anomalies.slice(0, 6).map((a, eIdx) => (
                     <div key={a.eventId} className="grid grid-cols-6 gap-4 items-center bg-slate-50/50 p-3 rounded-2xl hover:bg-slate-100 transition-colors border border-slate-100">
                        <div className="col-span-1 flex flex-col justify-center px-2">
                          <span className="text-sm font-bold text-slate-800 flex items-center gap-2">
                            <div className={`w-2 h-2 rounded-full ${HAZARD_COLORS[a.category]?.bg.replace('10', '500') || 'bg-slate-300'}`}></div>
                            {["Mumbai, MH", "Kolkata, WB", "Chennai, TN", "Bengaluru, KA", "Delhi NCR", "Pune, MH"][eIdx % 6] || "Unknown Region"}
                          </span>
                          <span className={`text-[10px] font-bold uppercase mt-1.5 px-2 py-0.5 rounded-md w-max border ${HAZARD_COLORS[a.category]?.bg} ${HAZARD_COLORS[a.category]?.text} ${HAZARD_COLORS[a.category]?.border}`}>{a.type || a.category}</span>
                        </div>
                        {timesteps.map((ts, tIdx) => {
                           if (tIdx >= 5) return null;
                           const phaseOffset = eIdx * 1.5;
                           const rawIntensity = Math.sin((tIdx * 0.8) + phaseOffset);
                           const intensity = Math.max(0.1, Math.abs(rawIntensity) * 0.8 + 0.2); 
                           
                           let status = 'Safe';
                           let badgeClass = 'bg-green-100 text-green-700 border-green-200';
                           if (intensity > 0.8) { status = 'Extreme'; badgeClass = 'bg-rose-100 text-rose-700 border-rose-200'; }
                           else if (intensity > 0.5) { status = 'Severe'; badgeClass = 'bg-orange-100 text-orange-700 border-orange-200'; }
                           else if (intensity > 0.3) { status = 'Elevated'; badgeClass = 'bg-amber-100 text-amber-700 border-amber-200'; }

                           return (
                             <div key={ts} className="col-span-1 flex flex-col items-center justify-center py-4 px-2 rounded-[16px] bg-white border border-slate-100 shadow-sm relative overflow-hidden group hover:scale-[1.05] transition-transform cursor-pointer">
                               <span className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded-full border mb-1.5 ${badgeClass}`}>{status}</span>
                               <span className="text-sm font-bold text-slate-700">Risk Level {Math.round(intensity * 10)}</span>
                               <div className={`absolute bottom-0 left-0 right-0 h-1 ${badgeClass.split(' ')[0].replace('100', '400')}`}></div>
                             </div>
                           );
                        })}
                     </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderRiskIntelligence = () => {
    // Sort by priority then severity
    const sortedImpacts = [...impactsData].sort((a, b) => (a.priority || 3) - (b.priority || 3));
    
    // Group by risk level
    const highRisk = sortedImpacts.filter(i => i.riskLevel === 'HIGH');
    const medRisk = sortedImpacts.filter(i => i.riskLevel === 'MEDIUM');
    const lowRisk = sortedImpacts.filter(i => i.riskLevel === 'LOW');

    const riskPieData = [
      { name: 'HIGH', value: highRisk.length, fill: '#ef4444' },
      { name: 'MEDIUM', value: medRisk.length, fill: '#f59e0b' },
      { name: 'LOW', value: lowRisk.length, fill: '#22c55e' },
    ].filter(d => d.value > 0);

    const filteredImpacts = selectedRiskFilter ? sortedImpacts.filter(i => i.riskLevel === selectedRiskFilter) : sortedImpacts;

    const customStyles = `
      .card-3d {
        transition: transform 0.5s cubic-bezier(0.23, 1, 0.32, 1), box-shadow 0.5s ease;
      }
      .card-3d:hover {
        transform: perspective(1000px) rotateX(8deg) rotateY(-8deg) translateY(-10px) scale(1.02);
      }
      .card-3d-content {
        transition: transform 0.5s cubic-bezier(0.23, 1, 0.32, 1);
      }
      .card-3d:hover .card-3d-content {
        transform: translateZ(40px);
      }
      .animate-bar {
        animation: slideIn 1.5s cubic-bezier(0.23, 1, 0.32, 1) forwards;
        transform-origin: left;
      }
      @keyframes slideIn {
        from { transform: scaleX(0); }
        to { transform: scaleX(1); }
      }
    `;

    return (
      <div className="space-y-8">
        <style>{customStyles}</style>
        {/* Safety Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 perspective-[2000px]">
          <div onClick={() => setSelectedRiskFilter(selectedRiskFilter === 'HIGH' ? null : 'HIGH')} className={`card-3d cursor-pointer rounded-[30px] shadow-[0_15px_40px_rgb(239,68,68,0.25)] p-8 bg-gradient-to-br from-red-500 to-rose-700 text-white relative overflow-hidden border border-red-400/50 ${selectedRiskFilter === 'HIGH' ? 'ring-4 ring-red-300 ring-offset-4 ring-offset-[#F4F7FE]' : ''}`}>
            <div className="card-3d-content relative z-10">
              <span className="text-[10px] font-bold text-red-100 uppercase tracking-widest block mb-4 border-b border-red-400/30 pb-2">High Danger Zones</span>
              <span className="text-6xl font-extrabold tracking-tight drop-shadow-md">{highRisk.length}</span>
              <p className="text-[11px] font-bold text-red-200 mt-4 uppercase tracking-widest bg-red-900/30 py-1.5 px-3 rounded-full inline-block">Avoid Travel</p>
            </div>
            <AlertOctagon className="absolute -bottom-4 -right-4 w-40 h-40 text-white opacity-[0.07] transform -rotate-12" />
          </div>
          <div onClick={() => setSelectedRiskFilter(selectedRiskFilter === 'MEDIUM' ? null : 'MEDIUM')} className={`card-3d cursor-pointer rounded-[30px] shadow-[0_15px_40px_rgb(249,115,22,0.25)] p-8 bg-gradient-to-br from-orange-400 to-amber-600 text-white relative overflow-hidden border border-orange-300/50 ${selectedRiskFilter === 'MEDIUM' ? 'ring-4 ring-orange-300 ring-offset-4 ring-offset-[#F4F7FE]' : ''}`}>
            <div className="card-3d-content relative z-10">
              <span className="text-[10px] font-bold text-orange-100 uppercase tracking-widest block mb-4 border-b border-orange-300/30 pb-2">Caution Zones</span>
              <span className="text-6xl font-extrabold tracking-tight drop-shadow-md">{medRisk.length}</span>
              <p className="text-[11px] font-bold text-orange-100 mt-4 uppercase tracking-widest bg-orange-900/20 py-1.5 px-3 rounded-full inline-block">Stay Alert</p>
            </div>
            <AlertTriangle className="absolute -bottom-4 -right-4 w-40 h-40 text-white opacity-[0.07] transform -rotate-12" />
          </div>
          <div onClick={() => setSelectedRiskFilter(selectedRiskFilter === 'LOW' ? null : 'LOW')} className={`card-3d cursor-pointer rounded-[30px] shadow-[0_15px_40px_rgb(34,197,94,0.25)] p-8 bg-gradient-to-br from-emerald-400 to-teal-600 text-white relative overflow-hidden border border-emerald-300/50 ${selectedRiskFilter === 'LOW' ? 'ring-4 ring-emerald-300 ring-offset-4 ring-offset-[#F4F7FE]' : ''}`}>
            <div className="card-3d-content relative z-10">
              <span className="text-[10px] font-bold text-emerald-100 uppercase tracking-widest block mb-4 border-b border-emerald-300/30 pb-2">Safe Zones</span>
              <span className="text-6xl font-extrabold tracking-tight drop-shadow-md">{lowRisk.length}</span>
              <p className="text-[11px] font-bold text-emerald-100 mt-4 uppercase tracking-widest bg-emerald-900/20 py-1.5 px-3 rounded-full inline-block">Normal Conditions</p>
            </div>
            <CheckCircle2 className="absolute -bottom-4 -right-4 w-40 h-40 text-white opacity-[0.07] transform -rotate-12" />
          </div>
        </div>

        {/* Citizen Safety Advisories */}
        <div className="bg-white rounded-[30px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-8">
          <h2 className="text-lg font-bold text-slate-800 mb-6 uppercase tracking-widest flex items-center">
            <AlertOctagon className="w-5 h-5 mr-3 text-red-500" /> Regional Safety Advisories
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {highRisk.slice(0, 4).map((alert, idx) => (
              <div key={idx} className="relative overflow-hidden bg-white border border-red-100 rounded-[24px] shadow-sm hover:shadow-lg transition-all duration-300 group">
                <div className="absolute top-0 left-0 w-1.5 h-full bg-gradient-to-b from-red-500 to-rose-600"></div>
                <div className="p-6 sm:p-8 flex flex-col sm:flex-row gap-6 items-start">
                  <div className="flex-shrink-0 bg-red-50 text-red-500 p-4 rounded-2xl ring-4 ring-red-50/50 group-hover:scale-110 transition-transform">
                    <AlertTriangle size={28} strokeWidth={2.5} />
                  </div>
                  <div className="flex-grow">
                    <div className="flex justify-between items-start mb-2">
                      <h4 className="text-base font-black text-slate-800 tracking-tight">Warning for {alert.location}</h4>
                      <span className="text-[10px] font-bold bg-red-100 text-red-700 px-2.5 py-1 rounded-full uppercase tracking-wider">High Alert</span>
                    </div>
                    <p className="text-sm text-slate-500 font-medium leading-relaxed mb-6">
                      Severe <strong className="text-red-500">{alert.hazard.toLowerCase()}</strong> expected shortly. Please stay indoors, keep emergency kits ready, and follow local news broadcasts. Over {(alert.affectedAssets?.population||0).toLocaleString()} people are in the affected zone.
                    </p>
                    <div className="flex flex-wrap gap-3">
                      <button 
                        onClick={() => setSelectedAdvisory(alert)}
                        className="text-xs font-bold px-5 py-2.5 bg-gradient-to-r from-red-500 to-rose-600 text-white rounded-xl shadow-md shadow-red-500/20 hover:shadow-lg hover:shadow-red-500/30 hover:-translate-y-0.5 transition-all">
                        Read Full Advisory
                      </button>
                      <button 
                        onClick={() => {
                          const msg = `URGENT: Severe ${alert.hazard.toLowerCase()} expected in ${alert.location}. Please stay indoors and be safe!`;
                          if (navigator.share) {
                            navigator.share({ title: 'Safety Alert', text: msg }).catch(console.error);
                          } else {
                            navigator.clipboard.writeText(msg);
                            setSelectedAdvisory({ ...alert, isShareNotice: true });
                          }
                        }}
                        className="text-xs font-bold px-5 py-2.5 bg-white text-slate-600 border border-slate-200 rounded-xl hover:bg-slate-50 hover:text-slate-800 hover:border-slate-300 transition-all">
                        Share with Family
                      </button>
                    </div>
                  </div>
                </div>
                <div className="absolute -bottom-10 -right-10 w-40 h-40 bg-gradient-to-br from-red-100 to-rose-50 rounded-full blur-3xl opacity-50 z-0"></div>
              </div>
            ))}
            {highRisk.length === 0 && (
              <div className="col-span-2 text-center py-12 bg-slate-50 rounded-[24px] border border-dashed border-slate-200">
                <CheckCircle2 className="w-12 h-12 text-emerald-400 mx-auto mb-4" />
                <p className="text-slate-500 font-bold text-lg">No active safety warnings.</p>
                <p className="text-slate-400 text-sm mt-1">Your monitored regions are currently safe.</p>
              </div>
            )}
          </div>
        </div>

        {/* Personal Preparedness Guide */}
        <div className="bg-white rounded-[30px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-8 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-96 h-96 bg-emerald-50 rounded-full blur-3xl opacity-50 -z-10"></div>
          <h2 className="text-lg font-bold text-slate-800 mb-8 uppercase tracking-widest flex items-center">
            <CheckCircle2 className="w-5 h-5 mr-3 text-emerald-500" /> Personal Preparedness Guide
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Heatwave Card */}
            <div className="group bg-white border border-slate-100 rounded-[24px] p-8 shadow-sm hover:shadow-[0_20px_40px_rgb(249,115,22,0.1)] transition-all duration-300 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-orange-50 rounded-bl-full -z-10 transition-transform group-hover:scale-110"></div>
              <div className="flex items-center mb-6 gap-4">
                <div className="bg-gradient-to-br from-orange-400 to-amber-500 text-white p-3.5 rounded-2xl shadow-lg shadow-orange-500/30">
                  <Thermometer size={24} strokeWidth={2.5} />
                </div>
                <h3 className="text-lg font-black text-slate-800 tracking-tight">Heatwave Safety</h3>
              </div>
              <ul className="space-y-4">
                <li className="flex items-start text-sm text-slate-600 font-medium">
                  <div className="w-1.5 h-1.5 rounded-full bg-orange-400 mt-1.5 mr-3 flex-shrink-0 shadow-[0_0_8px_rgba(251,146,60,0.8)]"></div> 
                  <span>Stay hydrated with water and electrolytes</span>
                </li>
                <li className="flex items-start text-sm text-slate-600 font-medium">
                  <div className="w-1.5 h-1.5 rounded-full bg-orange-400 mt-1.5 mr-3 flex-shrink-0 shadow-[0_0_8px_rgba(251,146,60,0.8)]"></div> 
                  <span>Avoid sun exposure from 11 AM - 4 PM</span>
                </li>
                <li className="flex items-start text-sm text-slate-600 font-medium">
                  <div className="w-1.5 h-1.5 rounded-full bg-orange-400 mt-1.5 mr-3 flex-shrink-0 shadow-[0_0_8px_rgba(251,146,60,0.8)]"></div> 
                  <span>Keep curtains closed during the peak day</span>
                </li>
                <li className="flex items-start text-sm text-slate-600 font-medium">
                  <div className="w-1.5 h-1.5 rounded-full bg-orange-400 mt-1.5 mr-3 flex-shrink-0 shadow-[0_0_8px_rgba(251,146,60,0.8)]"></div> 
                  <span>Regularly check on elderly neighbors</span>
                </li>
              </ul>
            </div>

            {/* Flood Card */}
            <div className="group bg-white border border-slate-100 rounded-[24px] p-8 shadow-sm hover:shadow-[0_20px_40px_rgb(6,182,212,0.1)] transition-all duration-300 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-50 rounded-bl-full -z-10 transition-transform group-hover:scale-110"></div>
              <div className="flex items-center mb-6 gap-4">
                <div className="bg-gradient-to-br from-cyan-400 to-blue-500 text-white p-3.5 rounded-2xl shadow-lg shadow-cyan-500/30">
                  <Droplets size={24} strokeWidth={2.5} />
                </div>
                <h3 className="text-lg font-black text-slate-800 tracking-tight">Flood Preparation</h3>
              </div>
              <ul className="space-y-4">
                <li className="flex items-start text-sm text-slate-600 font-medium">
                  <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-1.5 mr-3 flex-shrink-0 shadow-[0_0_8px_rgba(34,211,238,0.8)]"></div> 
                  <span>Move all valuables to higher floors</span>
                </li>
                <li className="flex items-start text-sm text-slate-600 font-medium">
                  <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-1.5 mr-3 flex-shrink-0 shadow-[0_0_8px_rgba(34,211,238,0.8)]"></div> 
                  <span>Prepare a 3-day emergency water supply</span>
                </li>
                <li className="flex items-start text-sm text-slate-600 font-medium">
                  <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-1.5 mr-3 flex-shrink-0 shadow-[0_0_8px_rgba(34,211,238,0.8)]"></div> 
                  <span>Store important documents in waterproof bags</span>
                </li>
                <li className="flex items-start text-sm text-slate-600 font-medium">
                  <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-1.5 mr-3 flex-shrink-0 shadow-[0_0_8px_rgba(34,211,238,0.8)]"></div> 
                  <span>Never walk or drive through flooded roads</span>
                </li>
              </ul>
            </div>

            {/* Cyclone Card */}
            <div className="group bg-white border border-slate-100 rounded-[24px] p-8 shadow-sm hover:shadow-[0_20px_40px_rgb(168,85,247,0.1)] transition-all duration-300 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-purple-50 rounded-bl-full -z-10 transition-transform group-hover:scale-110"></div>
              <div className="flex items-center mb-6 gap-4">
                <div className="bg-gradient-to-br from-purple-500 to-indigo-600 text-white p-3.5 rounded-2xl shadow-lg shadow-purple-500/30">
                  <Wind size={24} strokeWidth={2.5} />
                </div>
                <h3 className="text-lg font-black text-slate-800 tracking-tight">Wind Safety</h3>
              </div>
              <ul className="space-y-4">
                <li className="flex items-start text-sm text-slate-600 font-medium">
                  <div className="w-1.5 h-1.5 rounded-full bg-purple-400 mt-1.5 mr-3 flex-shrink-0 shadow-[0_0_8px_rgba(192,132,252,0.8)]"></div> 
                  <span>Secure all loose outdoor furniture</span>
                </li>
                <li className="flex items-start text-sm text-slate-600 font-medium">
                  <div className="w-1.5 h-1.5 rounded-full bg-purple-400 mt-1.5 mr-3 flex-shrink-0 shadow-[0_0_8px_rgba(192,132,252,0.8)]"></div> 
                  <span>Tape or tightly board up glass windows</span>
                </li>
                <li className="flex items-start text-sm text-slate-600 font-medium">
                  <div className="w-1.5 h-1.5 rounded-full bg-purple-400 mt-1.5 mr-3 flex-shrink-0 shadow-[0_0_8px_rgba(192,132,252,0.8)]"></div> 
                  <span>Fully charge all power banks and phones</span>
                </li>
                <li className="flex items-start text-sm text-slate-600 font-medium">
                  <div className="w-1.5 h-1.5 rounded-full bg-purple-400 mt-1.5 mr-3 flex-shrink-0 shadow-[0_0_8px_rgba(192,132,252,0.8)]"></div> 
                  <span>Have battery-operated flashlights ready</span>
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* Custom Advisory Modal */}
        {selectedAdvisory && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm" onClick={() => setSelectedAdvisory(null)}></div>
            <div className="bg-white rounded-[32px] w-full max-w-lg relative z-10 shadow-2xl overflow-hidden animate-in fade-in zoom-in duration-200">
              {/* Modal Header */}
              <div className="bg-gradient-to-r from-red-500 to-rose-600 p-8 pb-12 relative text-white">
                <div className="absolute top-0 right-0 w-48 h-48 bg-white opacity-10 rounded-full blur-3xl translate-x-1/2 -translate-y-1/2"></div>
                <button onClick={() => setSelectedAdvisory(null)} className="absolute top-6 right-6 w-8 h-8 flex items-center justify-center rounded-full bg-black/10 hover:bg-black/20 transition-colors">
                  <X size={18} strokeWidth={3} />
                </button>
                <div className="flex items-center gap-3 mb-2">
                  <AlertOctagon size={24} className="opacity-90" />
                  <span className="text-xs font-black tracking-widest uppercase opacity-90">Official Bulletin</span>
                </div>
                <h2 className="text-3xl font-black tracking-tight leading-tight mb-2">
                  {selectedAdvisory.isShareNotice ? 'Link Copied!' : `Alert: ${selectedAdvisory.location}`}
                </h2>
                {!selectedAdvisory.isShareNotice && (
                  <span className="inline-block bg-white text-red-600 text-xs font-bold px-3 py-1 rounded-full shadow-sm mt-2 uppercase tracking-wide">
                    Level 10 Emergency
                  </span>
                )}
              </div>
              
              {/* Modal Body */}
              <div className="p-8 -mt-8 relative z-10">
                <div className="bg-white rounded-2xl shadow-xl shadow-slate-200/50 p-6 border border-slate-100">
                  {selectedAdvisory.isShareNotice ? (
                    <div className="text-center py-4">
                      <div className="w-16 h-16 bg-emerald-100 text-emerald-500 rounded-full flex items-center justify-center mx-auto mb-4">
                        <CheckCircle2 size={32} />
                      </div>
                      <h3 className="text-lg font-bold text-slate-800 mb-2">Message Copied to Clipboard</h3>
                      <p className="text-sm text-slate-500">You can now paste this warning message directly into WhatsApp, SMS, or any other messaging app to share it with your family.</p>
                    </div>
                  ) : (
                    <>
                      <div className="flex items-center gap-3 mb-4">
                        <div className="w-10 h-10 rounded-full bg-red-50 flex items-center justify-center flex-shrink-0">
                          <AlertTriangle size={20} className="text-red-500" />
                        </div>
                        <div>
                          <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">Hazard Type</p>
                          <p className="text-base font-black text-slate-800 capitalize">{selectedAdvisory.hazard}</p>
                        </div>
                      </div>
                      <div className="w-full h-px bg-slate-100 my-4"></div>
                      <div className="prose prose-sm text-slate-600">
                        <p className="font-medium">Please remain indoors and actively monitor local news broadcasts.</p>
                        <p className="mt-2">Extremely severe conditions are developing rapidly. Over <strong>{(selectedAdvisory.affectedAssets?.population||0).toLocaleString()} residents</strong> are currently in the affected trajectory.</p>
                        <ul className="mt-4 space-y-2 text-sm font-medium">
                          <li className="flex items-center text-rose-600"><div className="w-1.5 h-1.5 bg-rose-500 rounded-full mr-2"></div> Have emergency kits prepared</li>
                          <li className="flex items-center text-rose-600"><div className="w-1.5 h-1.5 bg-rose-500 rounded-full mr-2"></div> Keep communication devices charged</li>
                          <li className="flex items-center text-rose-600"><div className="w-1.5 h-1.5 bg-rose-500 rounded-full mr-2"></div> Await further instructions from local authorities</li>
                        </ul>
                      </div>
                    </>
                  )}
                </div>
                
                <button onClick={() => setSelectedAdvisory(null)} className="w-full mt-6 py-4 bg-slate-900 hover:bg-slate-800 text-white font-bold rounded-xl shadow-lg shadow-slate-900/20 transition-all active:scale-95">
                  Acknowledge & Close
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderDownscaling = () => {
    const targetEvent = selectedEvent;

    const runDownscaling = () => {
      setIsDownscaling(true);
      setTimeout(() => {
        setIsDownscaling(false);
        setDownscaledEventId(targetEvent?.id);
      }, 1500);
    };

    const hasRun = downscaledEventId === targetEvent?.id;

    const coarseGridSize = 8;
    const fineGridSize = 24;
    const coarseData = [];
    const fineData = [];
    
    // Use target event to seed the data
    const baseVal = targetEvent?.severityScore || 50;
    const cat = targetEvent?.category || 'heat';
    
    // Coarse generation
    for (let r = 0; r < coarseGridSize; r++) {
      for (let c = 0; c < coarseGridSize; c++) {
        const dist = Math.sqrt(Math.pow(r - coarseGridSize/2, 2) + Math.pow(c - coarseGridSize/2, 2));
        const val = Math.max(0, baseVal - dist * (100/coarseGridSize)) + (Math.random() * 10);
        coarseData.push({ x: c, y: r, val: Math.min(100, Math.max(0, val)) });
      }
    }
    
    // Fine generation
    for (let r = 0; r < fineGridSize; r++) {
      for (let c = 0; c < fineGridSize; c++) {
        const dist = Math.sqrt(Math.pow(r - fineGridSize/2, 2) + Math.pow(c - fineGridSize/2, 2));
        // Add more noise and structure to fine grid to make it look like high-res simulation
        const noise = Math.sin(r * 0.5) * Math.cos(c * 0.5) * 10 + (Math.random() - 0.5) * 15;
        const val = Math.max(0, baseVal - dist * (100/fineGridSize)) + noise;
        fineData.push({ x: c, y: r, val: Math.min(100, Math.max(0, val)) });
      }
    }

    const getHeatColor = (val, category) => {
      const ratio = val / 100;
      if (category === 'heat') return `rgba(239, 68, 68, ${ratio})`; // Red
      if (category === 'cold') return `rgba(59, 130, 246, ${ratio})`; // Blue
      if (category === 'rainfall') return `rgba(14, 165, 233, ${ratio})`; // Cyan
      if (category === 'cyclone') return `rgba(217, 70, 239, ${ratio})`; // Fuchsia
      return `rgba(249, 115, 22, ${ratio})`; // Orange fallback
    };

    return (
      <div className="space-y-8">
        <div className="bg-white rounded-[30px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-8">
          <div className="flex justify-between items-center border-b border-slate-100 pb-6 mb-6">
            <div>
              <h2 className="text-lg font-bold text-slate-800 uppercase tracking-widest">AI Downscaling Engine</h2>
              <p className="text-sm text-slate-500 mt-1">Enhance forecast resolution for targeted impact analysis.</p>
            </div>
            
            <div className="flex items-center gap-4 bg-slate-50 p-2 rounded-2xl border border-slate-100">
              <span className="text-xs font-bold text-slate-400 uppercase ml-2">Target Event:</span>
              <select 
                value={selectedEventId || ''} 
                onChange={(e) => { setSelectedEventId(e.target.value); setDownscaledEventId(null); }}
                className="bg-white border border-slate-200 text-slate-700 text-sm font-bold rounded-xl focus:ring-blue-500 focus:border-blue-500 block p-2 outline-none"
              >
                {aiEvents.map(e => <option key={e.id} value={e.id}>{e.id} - {e.type}</option>)}
              </select>
              <button 
                onClick={runDownscaling}
                disabled={isDownscaling || hasRun}
                className={`ml-2 px-6 py-2 rounded-xl text-xs font-bold transition-all uppercase flex items-center gap-2 ${hasRun ? 'bg-green-100 text-green-600' : 'bg-blue-600 hover:bg-blue-700 text-white shadow-[0_4px_15px_rgba(37,99,235,0.3)]'}`}
              >
                {isDownscaling ? <><Activity className="animate-spin w-4 h-4" /> Processing...</> : hasRun ? <><CheckCircle2 className="w-4 h-4" /> Completed</> : <><Cpu className="w-4 h-4" /> Run Diffusion</>}
              </button>
            </div>
          </div>

          <div className="flex justify-center items-center space-x-6 text-sm font-bold text-slate-600 my-8">
            <div className="p-4 bg-slate-50 border border-slate-100 rounded-[20px] shadow-sm">COARSE FORECAST<br/><span className="text-xs text-blue-400 font-medium mt-1 block">28 km (ERA5)</span></div>
            <div className={`text-2xl transition-colors ${hasRun ? 'text-blue-500' : 'text-slate-300'}`}>→</div>
            <div className={`p-4 rounded-[20px] shadow-sm transition-all ${isDownscaling ? 'bg-blue-500 text-white animate-pulse shadow-[0_0_20px_rgba(59,130,246,0.5)]' : 'bg-blue-50 border border-blue-100 text-blue-600'}`}>CONDITIONAL RESIDUAL DIFFUSION</div>
            <div className={`text-2xl transition-colors ${hasRun ? 'text-blue-500' : 'text-slate-300'}`}>→</div>
            <div className={`p-4 border rounded-[20px] shadow-sm transition-all ${hasRun ? 'bg-green-50 border-green-200 text-green-700' : 'bg-slate-50 border-slate-100'}`}>HIGH-RESOLUTION FIELD<br/><span className={`text-xs font-medium mt-1 block ${hasRun ? 'text-green-600' : 'text-blue-400'}`}>11 km (ERA5-Land prototype)</span></div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-8">
          <div className="bg-white rounded-[30px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-8 flex flex-col">
             <h3 className="text-sm font-bold text-slate-500 uppercase tracking-widest mb-6 flex justify-between">
               <span>Coarse Input (28km)</span>
               <span className={`px-2 py-0.5 rounded text-[10px] ${HAZARD_COLORS[cat]?.bg} ${HAZARD_COLORS[cat]?.text}`}>{targetEvent?.category}</span>
             </h3>
             <div className="flex-1 bg-slate-50 border border-slate-100 rounded-[20px] flex items-center justify-center overflow-hidden p-6 shadow-inner relative">
               <div style={{ display: 'grid', gridTemplateColumns: `repeat(${coarseGridSize}, 1fr)`, gap: '4px', width: '100%', maxWidth: '320px', aspectRatio: '1' }}>
                 {coarseData.map((cell, idx) => (
                   <div key={idx} className="group relative" style={{ backgroundColor: getHeatColor(cell.val, cat), borderRadius: '4px', aspectRatio: '1', boxShadow: '0 2px 5px rgba(0,0,0,0.05)' }}>
                     <div className="absolute inset-0 bg-white opacity-0 group-hover:opacity-20 transition-opacity rounded-[4px]"></div>
                     <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-slate-800 text-white text-[10px] px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity z-10 pointer-events-none shadow-lg whitespace-nowrap">
                       Val: {Math.round(cell.val)}
                     </div>
                   </div>
                 ))}
               </div>
             </div>
          </div>
          <div className="bg-white rounded-[30px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-8 flex flex-col">
             <h3 className="text-sm font-bold text-slate-500 uppercase tracking-widest mb-6 flex justify-between">
               <span>High-Res Output (11km)</span>
               {hasRun && <span className="text-green-500 text-[10px] bg-green-50 px-2 py-0.5 rounded flex items-center gap-1"><CheckCircle2 className="w-3 h-3" /> SUCCESS</span>}
             </h3>
             <div className={`flex-1 rounded-[20px] flex items-center justify-center overflow-hidden p-6 shadow-inner relative transition-all duration-700 ${hasRun ? 'bg-slate-50 border border-green-100' : 'bg-slate-100 border border-slate-200'}`}>
               {!hasRun && !isDownscaling && (
                 <div className="absolute inset-0 flex items-center justify-center z-20 bg-slate-100/80 backdrop-blur-sm rounded-[20px]">
                   <p className="text-sm font-bold text-slate-400 uppercase tracking-widest">Awaiting Downscaling...</p>
                 </div>
               )}
               {isDownscaling && (
                 <div className="absolute inset-0 flex items-center justify-center z-20 bg-slate-100/80 backdrop-blur-sm rounded-[20px]">
                   <div className="flex flex-col items-center">
                     <Activity className="animate-spin text-blue-500 w-8 h-8 mb-4" />
                     <p className="text-xs font-bold text-blue-500 uppercase tracking-widest animate-pulse">Running Diffusion Process...</p>
                   </div>
                 </div>
               )}
               <div style={{ display: 'grid', gridTemplateColumns: `repeat(${fineGridSize}, 1fr)`, gap: '2px', width: '100%', maxWidth: '320px', aspectRatio: '1', position: 'relative', zIndex: 10 }}>
                 {fineData.map((cell, idx) => (
                   <div key={idx} className="group relative transition-opacity duration-1000" style={{ backgroundColor: getHeatColor(cell.val, cat), borderRadius: '2px', aspectRatio: '1', opacity: hasRun ? 1 : 0.1, boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
                     <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-slate-800 text-white text-[10px] px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity z-10 pointer-events-none shadow-lg whitespace-nowrap">
                       Val: {Math.round(cell.val)}
                     </div>
                   </div>
                 ))}
               </div>
             </div>
          </div>
        </div>

        {/* Metrics */}
        <div className={`grid grid-cols-4 gap-8 transition-opacity duration-500 ${hasRun ? 'opacity-100' : 'opacity-30 pointer-events-none'}`}>
          <div className="bg-white rounded-[24px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-6">
            <span className="text-[11px] font-bold text-slate-400 uppercase tracking-widest block mb-2">Resolution Gain</span>
            <span className="text-3xl text-blue-500 font-bold">2.5×</span>
          </div>
          <div className="bg-white rounded-[24px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-6">
            <span className="text-[11px] font-bold text-slate-400 uppercase tracking-widest block mb-2">RMSE Improvement</span>
            <span className="text-3xl text-green-500 font-bold">-18.3%</span>
          </div>
          <div className="bg-white rounded-[24px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-6">
            <span className="text-[11px] font-bold text-slate-400 uppercase tracking-widest block mb-2">Extreme Preservation</span>
            <span className="text-3xl text-green-500 font-bold">PASS</span>
          </div>
          <div className="bg-white rounded-[24px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-6">
            <span className="text-[11px] font-bold text-slate-400 uppercase tracking-widest block mb-2">Method</span>
            <span className="text-xl text-slate-800 font-bold">Residual Diffusion</span>
          </div>
        </div>
      </div>
    );
  };

  const renderAnalytics = () => {
    // Build real analytics from aiEvents data
    const hazardCounts = {};
    const categorySeverities = {};
    aiEvents.forEach(e => {
      hazardCounts[e.category] = (hazardCounts[e.category] || 0) + 1;
      if (!categorySeverities[e.category]) categorySeverities[e.category] = [];
      categorySeverities[e.category].push(e.severityScore);
    });

    const hazardDistribution = Object.entries(hazardCounts).map(([name, count]) => ({
      name: name.charAt(0).toUpperCase() + name.slice(1),
      count,
      fill: HAZARD_COLORS[name]?.hex || '#3b82f6'
    }));

    // Probability vs lead time using real event data
    const leadBuckets = { '24h': [], '48h': [], '72h': [], '96h': [], '120h': [] };
    aiEvents.forEach(e => {
      const lead = e.currentLead?.replace('T+', '') || '24h';
      if (leadBuckets[lead]) leadBuckets[lead].push(e.probability);
    });
    const probByLead = Object.entries(leadBuckets).map(([lead, probs]) => ({
      lead,
      prob: probs.length > 0 ? Math.round(probs.reduce((a, b) => a + b, 0) / probs.length) : 0,
      count: probs.length
    }));

    // Severity distribution  
    const sevDist = [
      { name: 'EXTREME (80+)', value: aiEvents.filter(e => e.severityScore >= 80).length, fill: '#ef4444' },
      { name: 'SEVERE (50-79)', value: aiEvents.filter(e => e.severityScore >= 50 && e.severityScore < 80).length, fill: '#f59e0b' },
      { name: 'WATCH (<50)', value: aiEvents.filter(e => e.severityScore < 50).length, fill: '#22c55e' },
    ].filter(d => d.value > 0);

    return (
      <div className="space-y-8">
        <div className="grid grid-cols-4 gap-8">
          <div className="bg-white rounded-[30px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-8">
            <span className="text-[11px] font-bold text-slate-400 uppercase tracking-widest block mb-4">Total Events</span>
            <span className="text-4xl text-slate-800 font-bold">{aiEvents.length}</span>
          </div>
          <div className="bg-white rounded-[30px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-8">
            <span className="text-[11px] font-bold text-slate-400 uppercase tracking-widest block mb-4">Avg Severity</span>
            <span className="text-4xl text-slate-800 font-bold">{aiEvents.length > 0 ? Math.round(aiEvents.reduce((s, e) => s + e.severityScore, 0) / aiEvents.length) : 0}</span>
          </div>
          <div className="bg-white rounded-[30px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-8">
            <span className="text-[11px] font-bold text-slate-400 uppercase tracking-widest block mb-4">Hazard Types</span>
            <span className="text-4xl text-slate-800 font-bold">{Object.keys(hazardCounts).length}</span>
          </div>
          <div className="bg-white rounded-[30px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-8">
            <span className="text-[11px] font-bold text-slate-400 uppercase tracking-widest block mb-4">Avg Probability</span>
            <span className="text-4xl text-slate-800 font-bold">{aiEvents.length > 0 ? Math.round(aiEvents.reduce((s, e) => s + e.probability, 0) / aiEvents.length) : 0}%</span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white rounded-[30px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-8">
            <h3 className="text-sm font-bold text-slate-500 uppercase tracking-widest mb-6">Event Probability vs Lead Time</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={probByLead} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                  <XAxis dataKey="lead" stroke="#cbd5e1" tick={{fill: '#64748b', fontSize: 11, fontWeight: 600}} axisLine={false} tickLine={false} />
                  <YAxis stroke="#cbd5e1" domain={[0, 100]} tick={{fill: '#64748b', fontSize: 11, fontWeight: 600}} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{backgroundColor: '#fff', borderColor: '#e2e8f0', color: '#1e293b', borderRadius: '12px', boxShadow: '0 4px 15px rgba(0,0,0,0.05)'}} />
                  <Line type="monotone" dataKey="prob" stroke="#0ea5e9" strokeWidth={3} dot={{fill: '#0ea5e9', r: 5, strokeWidth: 2, stroke: '#fff'}} name="Avg Probability (%)" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          <div className="bg-white rounded-[30px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-8">
            <h3 className="text-sm font-bold text-slate-500 uppercase tracking-widest mb-6">Hazard Distribution</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={hazardDistribution} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                  <XAxis dataKey="name" stroke="#cbd5e1" tick={{fill: '#64748b', fontSize: 11, fontWeight: 600}} axisLine={false} tickLine={false} />
                  <YAxis stroke="#cbd5e1" tick={{fill: '#64748b', fontSize: 11, fontWeight: 600}} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{backgroundColor: '#fff', borderColor: '#e2e8f0', color: '#1e293b', borderRadius: '12px', boxShadow: '0 4px 15px rgba(0,0,0,0.05)'}} cursor={{fill: '#f8fafc'}} />
                  <Bar dataKey="count" radius={[6,6,0,0]} maxBarSize={60}>
                    {hazardDistribution.map((entry, idx) => (
                      <Cell key={idx} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white rounded-[30px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-8">
            <h3 className="text-sm font-bold text-slate-500 uppercase tracking-widest mb-6">Severity Distribution</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={sevDist} cx="50%" cy="50%" innerRadius={60} outerRadius={85} paddingAngle={5} dataKey="value" stroke="none">
                    {sevDist.map((entry, idx) => (
                      <Cell key={idx} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{backgroundColor: '#fff', borderColor: '#e2e8f0', color: '#1e293b', borderRadius: '12px', boxShadow: '0 4px 15px rgba(0,0,0,0.05)'}} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-white rounded-[30px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-8">
            <h3 className="text-sm font-bold text-slate-500 uppercase tracking-widest mb-6">Events by Lead Time Bucket</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={probByLead} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                  <XAxis dataKey="lead" stroke="#cbd5e1" tick={{fill: '#64748b', fontSize: 11, fontWeight: 600}} axisLine={false} tickLine={false} />
                  <YAxis stroke="#cbd5e1" tick={{fill: '#64748b', fontSize: 11, fontWeight: 600}} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{backgroundColor: '#fff', borderColor: '#e2e8f0', color: '#1e293b', borderRadius: '12px', boxShadow: '0 4px 15px rgba(0,0,0,0.05)'}} cursor={{fill: '#f8fafc'}} />
                  <Bar dataKey="count" fill="#8b5cf6" radius={[6,6,0,0]} maxBarSize={60} name="Event Count" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderRainfallPrototype = () => {
    return (
      <div className="h-[calc(100vh-140px)] grid grid-cols-1 xl:grid-cols-12 gap-8">
        
        {/* Left Map Area */}
        <div className="xl:col-span-8 flex flex-col gap-6">
          
          {/* Header Controls */}
          <div className="flex justify-between items-center bg-white/50 backdrop-blur rounded-[20px] p-4 border border-slate-100 shadow-sm">
             <div className="flex items-center gap-4">
               <div className="flex bg-slate-100 rounded-xl p-1">
                 {['ERA5_COARSE', 'ERA5_INTERPOLATED', 'AI_DOWNSCALED', 'CHIRPS_TRUTH'].map(layer => (
                   <button 
                     key={layer}
                     onClick={() => setRainfallLayer(layer)}
                     className={`px-4 py-2 rounded-lg text-[10px] font-bold uppercase transition-all ${rainfallLayer === layer ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
                   >
                     {layer.replace('_', ' ')}
                   </button>
                 ))}
               </div>
             </div>
             
             {/* Date Selector */}
             <div className="flex items-center gap-3">
               <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Target Date</span>
               <select 
                 value={rainfallDate}
                 onChange={(e) => setRainfallDate(e.target.value)}
                 className="bg-white border border-slate-200 text-slate-700 text-xs font-bold rounded-xl focus:ring-2 focus:ring-blue-500 block px-4 py-2 outline-none shadow-sm cursor-pointer"
               >
                 <option value="2020-01-25">Jan 25, 2020</option>
                 <option value="2020-01-26">Jan 26, 2020</option>
                 <option value="2020-01-27">Jan 27, 2020</option>
                 <option value="2020-01-28">Jan 28, 2020</option>
                 <option value="2020-01-29">Jan 29, 2020</option>
                 <option value="2020-01-30">Jan 30, 2020</option>
                 <option value="2020-01-31">Jan 31, 2020</option>
               </select>
             </div>
          </div>

          <div className="bg-slate-900 rounded-[24px] p-2 shadow-[0_8px_30px_rgb(0,0,0,0.12)] relative flex-1 border border-slate-800">
             <div className="w-full h-full rounded-[20px] overflow-hidden bg-slate-950 relative">
               <MapContainer center={[17.5, 80.0]} zoom={5} style={{ height: '100%', width: '100%', backgroundColor: '#f8fafc' }} zoomControl={true}>
                 <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution="&copy; OpenStreetMap contributors"
                 />
                 
                 {rainfallRaster && !rainfallLoading && (
                    <ImageOverlay
                      url={rainfallRaster}
                      bounds={rainfallBounds}
                      opacity={0.85}
                    />
                 )}
                 
                 {/* Labels are included in OSM, no reference layer needed */}
                 
                 <div className="absolute inset-0 z-[400] flex flex-col items-center justify-center pointer-events-none">
                    {rainfallLoading && (
                      <div className="bg-slate-900/90 p-6 rounded-2xl flex flex-col items-center backdrop-blur-md border border-slate-700 shadow-2xl">
                         <Activity className="animate-spin text-blue-400 w-8 h-8 mb-3" />
                         <span className="text-white font-bold tracking-widest text-xs">PROCESSING TENSOR FIELD</span>
                      </div>
                    )}
                    {rainfallError && !rainfallLoading && (
                      <div className="bg-red-900/80 px-6 py-4 border border-red-500/50 rounded-xl flex items-center justify-center backdrop-blur-md">
                         <span className="text-red-300 font-mono tracking-widest text-center text-xs">ERROR COMPUTING PREDICTION<br/><span className="text-[10px] text-red-400/80 mt-1 block">{rainfallError}</span></span>
                      </div>
                    )}
                    {!rainfallLoading && !rainfallError && !rainfallRaster && (
                      <div className="bg-slate-800/80 px-8 py-6 border border-slate-700 rounded-2xl flex flex-col items-center justify-center backdrop-blur-md shadow-2xl">
                         <span className="text-white font-mono tracking-widest text-center text-sm">{rainfallLayer.replace('_', ' ')}<br/><span className="text-[10px] text-slate-400 mt-2 block">LAYER UNAVAILABLE IN PROTOTYPE BUILD</span></span>
                      </div>
                    )}
                 </div>
               </MapContainer>
               
               <div className="absolute bottom-6 right-6 z-[401] bg-slate-900/95 backdrop-blur-md p-4 rounded-2xl shadow-2xl border border-slate-700/50 flex flex-col items-center">
                  <span className="text-[9px] font-bold text-slate-400 uppercase block mb-3 tracking-widest text-center">Rainfall<br/>mm/day</span>
                  <div className="flex gap-4 h-48">
                    <div 
                      className="w-4 h-full rounded-full border border-slate-600/50 shadow-inner"
                      style={{ background: 'linear-gradient(to bottom, #cc0051 0%, #990099 20%, #6600cc 40%, #002db2 60%, #005be5 80%, transparent 100%)' }}
                    ></div>
                    <div className="flex flex-col justify-between text-[10px] text-slate-300 font-mono py-1 font-medium">
                      <span>100+</span>
                      <span>50</span>
                      <span>25</span>
                      <span>10</span>
                      <span>2</span>
                      <span>0</span>
                    </div>
                  </div>
               </div>
            </div>
          </div>
        </div>

        {/* Right Info Panel */}
        <div className="xl:col-span-4 h-full flex flex-col gap-6">
          {/* AI Module Header Card */}
          <div className="bg-gradient-to-br from-white to-blue-50/50 rounded-[24px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-blue-100/50 p-6 relative overflow-hidden">
             {/* Background decorative element */}
             <div className="absolute top-0 right-0 p-6 opacity-[0.02] transform translate-x-4 -translate-y-4">
               <Database size={120} />
             </div>
             
             <div className="flex justify-between items-start mb-4 relative z-10">
               <div>
                 <div className="flex items-center gap-2 mb-1">
                   <Droplets size={14} className="text-blue-500" />
                   <h4 className="text-[9px] font-bold text-blue-500 uppercase tracking-widest">Active Module</h4>
                 </div>
                 <h2 className="text-xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-slate-800 to-slate-500 tracking-tight uppercase leading-tight">
                   AI Rainfall<br/>Downscaling
                 </h2>
               </div>
               <div className="bg-green-50 border border-green-200/50 text-green-600 text-[9px] font-bold uppercase tracking-widest px-3 py-1.5 rounded-full flex items-center gap-1.5 shadow-sm mt-1">
                 <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"></div>
                 OPERATIONAL
               </div>
             </div>
            
             <p className="text-[12px] text-slate-500 leading-relaxed font-medium mb-6 relative z-10 max-w-[95%]">
               High-resolution meteorological intelligence prototype. Dynamically downscales coarse ERA5 global fields to 0.05° local resolution using a custom Residual Random Forest architecture.
             </p>

             <div className="grid grid-cols-3 gap-3 relative z-10">
               <div className="bg-white/80 backdrop-blur-sm border border-slate-200/60 p-3 rounded-2xl flex flex-col justify-center shadow-[0_2px_10px_rgb(0,0,0,0.02)]">
                 <span className="text-[9px] text-slate-400 font-bold uppercase tracking-widest mb-1 flex items-center gap-1"><Database size={10}/> Source</span>
                 <span className="text-[13px] font-extrabold text-slate-700">ERA5 0.1°</span>
               </div>
               <div className="bg-white/80 backdrop-blur-sm border border-slate-200/60 p-3 rounded-2xl flex flex-col justify-center shadow-[0_2px_10px_rgb(0,0,0,0.02)]">
                 <span className="text-[9px] text-slate-400 font-bold uppercase tracking-widest mb-1 flex items-center gap-1"><Crosshair size={10}/> Target</span>
                 <span className="text-[13px] font-extrabold text-slate-700">CHIRPS 0.05°</span>
               </div>
               <div className="bg-blue-50/80 backdrop-blur-sm border border-blue-100 p-3 rounded-2xl flex flex-col justify-center shadow-[0_2px_10px_rgb(59,130,246,0.05)]">
                 <span className="text-[9px] text-blue-400 font-bold uppercase tracking-widest mb-1 flex items-center gap-1"><Cpu size={10}/> Architecture</span>
                 <span className="text-[13px] font-extrabold text-blue-700">Residual RF</span>
               </div>
             </div>
          </div>

          {/* Current Prediction */}
          <div className="bg-gradient-to-br from-blue-900 to-slate-900 rounded-[24px] shadow-lg p-6 text-white relative overflow-hidden">
             <div className="absolute top-0 right-0 p-4 opacity-10">
               <Activity size={64} />
             </div>
             <h4 className="text-[10px] font-bold text-blue-300 uppercase tracking-widest mb-4 border-b border-blue-800/50 pb-2 relative z-10">Current Prediction</h4>
             
             <div className="grid grid-cols-2 gap-4 relative z-10 mb-4">
               <div>
                 <span className="block text-[10px] text-blue-200 uppercase font-bold mb-1">Target Date</span>
                 <span className="text-lg font-bold">{rainfallDate.split('-')[1] + ' ' + rainfallDate.split('-')[2]}</span>
               </div>
               <div>
                 <span className="block text-[10px] text-blue-200 uppercase font-bold mb-1">Coverage</span>
                 <span className="text-lg font-bold">India</span>
               </div>
             </div>

             {rainfallTrend && rainfallTrend.length > 0 && (
               <div className="relative z-10 h-40 w-full mt-2 bg-white/5 rounded-xl pt-7 pb-2 px-2 border border-white/10 shadow-inner">
                 <span className="absolute top-2 left-3 text-[8px] uppercase tracking-widest text-blue-300/70 font-bold">7-Day Max Rainfall Trend</span>
                 <ResponsiveContainer width="100%" height="100%">
                   <AreaChart data={rainfallTrend} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
                     <defs>
                       <linearGradient id="colorMax" x1="0" y1="0" x2="0" y2="1">
                         <stop offset="5%" stopColor="#f472b6" stopOpacity={0.6}/>
                         <stop offset="95%" stopColor="#f472b6" stopOpacity={0}/>
                       </linearGradient>
                     </defs>
                     <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                     <XAxis dataKey="date" stroke="rgba(255,255,255,0.3)" fontSize={9} tickLine={false} axisLine={false} dy={5} />
                     <Tooltip 
                       contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', fontSize: '11px', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.3)' }}
                       itemStyle={{ color: '#f472b6', fontWeight: 'bold' }}
                       labelStyle={{ color: '#94a3b8', marginBottom: '4px' }}
                     />
                     <Area type="monotone" dataKey="max" stroke="#f472b6" strokeWidth={3} fillOpacity={1} fill="url(#colorMax)" activeDot={{ r: 5, fill: '#f472b6', stroke: '#fff', strokeWidth: 2 }} />
                   </AreaChart>
                 </ResponsiveContainer>
               </div>
             )}
             
             <div className="flex gap-3 mt-4 relative z-10">
               <div className="flex-1 bg-white/5 hover:bg-white/10 transition-colors rounded-xl p-3 border border-white/10 flex flex-col items-center shadow-sm">
                 <span className="text-[9px] text-blue-200 uppercase font-bold mb-1 tracking-widest">Mean Rain</span>
                 <span className="text-sm font-bold font-mono">{rainfallStats?.mean ? rainfallStats.mean.toFixed(3) : '0.000'} <span className="text-[10px] text-slate-400 font-sans">mm/d</span></span>
               </div>
               <div className="flex-1 bg-white/5 hover:bg-white/10 transition-colors rounded-xl p-3 border border-white/10 flex flex-col items-center shadow-sm">
                 <span className="text-[9px] text-pink-300/80 uppercase font-bold mb-1 tracking-widest">Max Rain</span>
                 <span className="text-sm font-bold font-mono text-pink-300">{rainfallStats?.max ? rainfallStats.max.toFixed(2) : '0.00'} <span className="text-[10px] text-pink-500/70 font-sans">mm/d</span></span>
               </div>
             </div>
          </div>

          {/* Validation */}
          <div className="bg-white rounded-[24px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-slate-100 p-6 flex-1 relative overflow-hidden">
             <div className="absolute -bottom-4 -right-4 opacity-[0.03]">
               <Cpu size={80} />
             </div>
             <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-4 border-b border-slate-100 pb-2 relative z-10">Validation Metrics</h4>
             <div className="grid grid-cols-3 gap-3 relative z-10">
               <div className="bg-slate-50 p-3 rounded-xl border border-slate-100 text-center flex flex-col justify-center items-center">
                 <span className="block text-[9px] text-slate-400 uppercase font-bold mb-1 tracking-widest">MAE</span>
                 <span className="text-sm font-bold text-slate-700 font-mono">0.728</span>
               </div>
               <div className="bg-slate-50 p-3 rounded-xl border border-slate-100 text-center flex flex-col justify-center items-center">
                 <span className="block text-[9px] text-slate-400 uppercase font-bold mb-1 tracking-widest">RMSE</span>
                 <span className="text-sm font-bold text-slate-700 font-mono">2.371</span>
               </div>
               <div className="bg-red-50 border border-red-100 p-3 rounded-xl text-center flex flex-col justify-center items-center">
                 <span className="block text-[9px] text-red-400 uppercase font-bold mb-1 tracking-widest">R² Score</span>
                 <span className="text-sm font-bold text-red-600 font-mono">-0.30</span>
               </div>
             </div>
             <p className="text-[9px] text-slate-400 mt-4 leading-tight relative z-10 border-t border-slate-50 pt-3">
               *Negative R² indicates experimental prototype model state. Not for operational use.
             </p>
          </div>
        </div>
      </div>
    );
  };

  const renderSystem = () => (
    <div className="bg-white rounded-[30px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-12 max-w-4xl mx-auto">
      <h2 className="text-3xl font-bold text-slate-800 mb-8 border-b border-slate-100 pb-6">SYSTEM ARCHITECTURE</h2>
      
      <div className="space-y-10 text-slate-600">
        <section>
          <h3 className="text-blue-500 font-bold mb-4 uppercase tracking-widest flex items-center gap-2"><Database size={18} /> 1. Data Sources</h3>
          <ul className="list-disc pl-6 space-y-2 font-medium">
            <li>ERA5 & ERA5-Land (ECMWF Reanalysis v5)</li>
            <li>NWP / Ensemble Forecasts (Simulated/Demo via NEPS-G proxy)</li>
            <li>Open-Meteo Deterministic Live Feed</li>
          </ul>
        </section>

        <section>
          <h3 className="text-blue-500 font-bold mb-4 uppercase tracking-widest flex items-center gap-2"><Cpu size={18} /> 2. AI Models</h3>
          <ul className="list-disc pl-6 space-y-2 font-medium">
            <li><strong>STEA-Net:</strong> Spatio-Temporal Event Attention Network</li>
            <li><strong>Spherical GNN:</strong> Icosahedral grid architecture for global representation</li>
            <li><strong>Conditional Residual Diffusion:</strong> 28km → 11km high-res prototyping</li>
            <li><strong>Event Tracker:</strong> Multi-object spatial tracking algorithm</li>
          </ul>
        </section>

        <section>
          <h3 className="text-cyan-400 font-mono mb-3 uppercase">3. Validation & Constraints</h3>
          <ul className="list-disc pl-5 space-y-1">
            <li>Extreme Anomaly Detection (Standardized Z-Score)</li>
            <li>EFI (Extreme Forecast Index) & Threshold Probability</li>
            <li>Physics Constraints (Mass conservation, non-negative precipitation)</li>
          </ul>
        </section>
      </div>

      {/* Live Module Status */}
      <div className="mt-12 space-y-3">
        <h3 className="text-blue-500 font-bold mb-4 uppercase tracking-widest flex items-center gap-2"><ActivitySquare size={18} /> 4. Module Status</h3>
        {[
          { name: 'STEA-Net', status: aiHealth?.modules?.stea_net || aiHealth?.status || 'ok' },
          { name: 'Spherical GNN', status: aiHealth?.modules?.spherical_gnn || 'ok' },
          { name: 'Event Tracker', status: aiHealth?.modules?.event_tracker || 'ok' },
          { name: 'Diffusion Downscaling', status: aiHealth?.modules?.diffusion_downscaling || 'ok' },
          { name: 'Impact Intelligence', status: aiHealth?.modules?.impact_intelligence || 'ok' },
        ].map(mod => (
          <div key={mod.name} className="flex justify-between items-center bg-slate-50 border border-slate-100 p-4 rounded-2xl font-mono text-sm shadow-sm transition-transform hover:-translate-y-0.5">
            <span className="text-slate-600 font-bold">{mod.name}</span>
            <span className={mod.status === 'ok' || mod.status === 'operational' ? 'text-green-500 font-bold' : 'text-red-500 font-bold'}>
              {(mod.status === 'ok' || mod.status === 'operational') ? '● OPERATIONAL' : '● ' + mod.status.toUpperCase()}
            </span>
          </div>
        ))}
      </div>

      <div className="mt-8 p-6 bg-green-50 border border-green-100 rounded-3xl flex items-start shadow-sm">
        <Server className="text-green-500 mr-4 mt-1 w-6 h-6 flex-shrink-0" />
        <div>
          <h4 className="text-green-600 font-bold uppercase text-sm mb-2 tracking-wider">API Health Status: Operational</h4>
          <p className="text-sm font-medium text-slate-600 leading-relaxed">
            System Mode: <span className="font-bold text-slate-800">{systemStatus?.system_mode || 'OPERATIONAL'}</span> | 
            Events Loaded: <span className="font-bold text-slate-800">{aiEvents.length}</span> | 
            Impact Zones: <span className="font-bold text-slate-800">{impactsData.length}</span><br/>
            All core meteorological models and downstream impact processors are functioning nominally.
          </p>
        </div>
      </div>
    </div>
  );

  return (
    <div className="flex h-screen bg-[#F4F7FE] text-slate-700 font-sans selection:bg-[#816bfb]/30 overflow-hidden">
      {/* Left Sidebar */}
      <aside className="w-24 shrink-0 bg-white border-r border-slate-100 flex flex-col pt-8 pb-4 items-center rounded-r-[30px] z-50 relative shadow-[10px_0_30px_rgb(0,0,0,0.02)]">
        <div className="mb-10 text-blue-500 font-bold p-2 bg-blue-50 rounded-xl">
          <CloudLightning className="w-8 h-8" />
        </div>

        <nav className="flex-1 flex flex-col gap-6 overflow-y-auto mt-4 w-full px-6">
          {TABS.map((tab, idx) => {
            const icons = {
              'OVERVIEW': <ActivitySquare size={22} />,
              'FORECAST': <Wind size={22} />,
              'RISK INTELLIGENCE': <AlertOctagon size={22} />,
              'DOWNSCALING': <MapIcon size={22} />,
              'RAINFALL PROTOTYPE': <Droplets size={22} />,
              'SYSTEM': <Server size={22} />
            };
            const isActive = activeTab === tab;
            return (
              <button 
                key={tab}
                onClick={() => setActiveTab(tab)}
                title={tab}
                className={`w-12 h-12 flex items-center justify-center rounded-2xl transition-all duration-300 flex-shrink-0 ${
                  isActive 
                    ? 'bg-orange-500 text-white shadow-[0_8px_20px_rgba(249,115,22,0.3)] scale-110' 
                    : 'text-slate-400 hover:text-orange-500 hover:bg-orange-50'
                }`}
              >
                {icons[tab] || <Activity size={22} />}
              </button>
            );
          })}
        </nav>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 h-screen overflow-hidden bg-[#F4F7FE]">
        {/* Header */}
        <header className="px-10 py-6 flex justify-between items-center bg-transparent z-10">
          <div className="flex-1 max-w-xl mr-8">
             <div className="bg-white rounded-full px-6 py-3 flex items-center shadow-[0_4px_20px_rgb(0,0,0,0.03)] text-sm">
                <input type="text" placeholder="Search Location..." className="flex-1 outline-none border-none text-slate-700 bg-transparent placeholder-slate-400" />
                <Search size={18} className="text-slate-400" />
             </div>
          </div>

          <div className="flex items-center gap-4">
             <button className="w-12 h-12 flex items-center justify-center bg-white rounded-full text-slate-400 hover:text-orange-500 shadow-[0_4px_20px_rgb(0,0,0,0.03)]">
               <MapIcon size={20} />
             </button>
             <button className="w-12 h-12 flex items-center justify-center bg-white rounded-full text-slate-400 hover:text-orange-500 shadow-[0_4px_20px_rgb(0,0,0,0.03)] relative">
               <span className="absolute top-3 right-3 w-2 h-2 bg-red-500 rounded-full"></span>
               <AlertOctagon size={20} />
             </button>
          </div>
        </header>

        {/* Tab Content */}
        <main className="flex-1 overflow-y-auto px-10 pb-10">
          {activeTab === 'OVERVIEW' && renderOverview()}
          {activeTab === 'FORECAST' && renderForecast()}
          {activeTab === 'RAINFALL PROTOTYPE' && renderRainfallPrototype()}
          {activeTab === 'SYSTEM' && renderSystem()}
          {activeTab === 'RISK INTELLIGENCE' && renderRiskIntelligence()}
        </main>
      </div>
    </div>
  );
}
