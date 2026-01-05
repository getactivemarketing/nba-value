import { Routes, Route } from 'react-router-dom';
import { Layout } from '@/components/Layout';
import { MarketBoard } from '@/pages/MarketBoard';
import { BetDetail } from '@/pages/BetDetail';
import { Trends } from '@/pages/Trends';
import { Evaluation } from '@/pages/Evaluation';

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<MarketBoard />} />
        <Route path="/bet/:marketId" element={<BetDetail />} />
        <Route path="/trends" element={<Trends />} />
        <Route path="/evaluation" element={<Evaluation />} />
      </Routes>
    </Layout>
  );
}

export default App;
