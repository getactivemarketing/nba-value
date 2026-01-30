import { Routes, Route } from 'react-router-dom';
import { Layout } from '@/components/Layout';
import { MarketBoard } from '@/pages/MarketBoard';
import { BetDetail } from '@/pages/BetDetail';
import { Trends } from '@/pages/Trends';
import { Evaluation } from '@/pages/Evaluation';
import { PlayerPropsPage } from '@/pages/PlayerPropsPage';

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<MarketBoard />} />
        <Route path="/bet/:marketId" element={<BetDetail />} />
        <Route path="/trends" element={<Trends />} />
        <Route path="/evaluation" element={<Evaluation />} />
        <Route path="/props" element={<PlayerPropsPage />} />
      </Routes>
    </Layout>
  );
}

export default App;
