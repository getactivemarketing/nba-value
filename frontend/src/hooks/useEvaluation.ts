import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { Algorithm } from '@/types/market';

export function useAlgorithmComparison(
  startDate?: string,
  endDate?: string,
  marketType?: string
) {
  return useQuery({
    queryKey: ['evaluation', 'compare', startDate, endDate, marketType],
    queryFn: () => api.getAlgorithmComparison(startDate, endDate, marketType),
    staleTime: 300000, // 5 minutes - evaluation data doesn't change often
  });
}

export function useCalibrationCurve(marketType?: string, algorithm: Algorithm = 'a') {
  return useQuery({
    queryKey: ['evaluation', 'calibration', marketType, algorithm],
    queryFn: () => api.getCalibrationCurve(marketType, algorithm),
    staleTime: 300000,
  });
}

export function usePerformanceByBucket(
  algorithm: Algorithm = 'a',
  bucketType: 'score' | 'edge' | 'confidence' = 'score'
) {
  return useQuery({
    queryKey: ['evaluation', 'performance', algorithm, bucketType],
    queryFn: () => api.getPerformanceByBucket(algorithm, bucketType),
    staleTime: 300000,
  });
}
