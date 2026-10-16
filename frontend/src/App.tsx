import { Navigate, Routes, Route } from 'react-router-dom';
import { Layout } from '@/components/Layout';
import { EdgeHome } from '@/pages/EdgeHome';
import { EdgeFinder } from '@/pages/EdgeFinder';
import { EdgeOdds } from '@/pages/EdgeOdds';
import { EdgePlayer } from '@/pages/EdgePlayer';
import { EdgeResults } from '@/pages/EdgeResults';
import { Trends } from '@/pages/Trends';
import { MLBPicks } from '@/pages/MLBPicks';
import { NRFI } from '@/pages/NRFI';
import { MLBEvaluation } from '@/pages/MLBEvaluation';
import { NFLPicks } from '@/pages/NFLPicks';
import { NFLEvaluation } from '@/pages/NFLEvaluation';

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<EdgeHome />} />
        <Route path="/edge" element={<EdgeFinder />} />
        <Route path="/edge/odds/:eventId" element={<EdgeOdds />} />
        <Route path="/edge/player/:id" element={<EdgePlayer />} />
        <Route path="/edge/results" element={<EdgeResults />} />
        <Route path="/mlb" element={<MLBPicks />} />
        <Route path="/mlb/performance" element={<MLBEvaluation />} />
        <Route path="/nrfi" element={<NRFI />} />
        <Route path="/nfl" element={<NFLPicks />} />
        <Route path="/nfl/performance" element={<NFLEvaluation />} />
        <Route path="/trends" element={<Trends />} />
        <Route path="/evaluation" element={<Navigate to="/edge/results" replace />} />
        <Route path="/props" element={<Navigate to="/edge" replace />} />
      </Routes>
    </Layout>
  );
}

export default App;
