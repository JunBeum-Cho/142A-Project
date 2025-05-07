'use client';

import { Title, Paper, SimpleGrid, Text, Center, Alert, Table, Button } from '@mantine/core';
import { BarChart, LineChart } from '@mantine/charts';
import resultsDataJson from '../public/result.json'; // Import JSON data directly
import './globals.css'; // 이 줄이 있는지 확인

interface ModelResult {
  Model: string;
  Accuracy?: number;
  Precision?: number;
  Recall?: number;
  'F1-Score'?: number;
  'ROC-AUC'?: number;
}

interface FeatureImportance {
  Feature: string;
  Importance: number;
}

interface RocCurveData {
  fpr: number[];
  tpr: number[];
  auc?: number;
}

interface ConfusionMatrixData {
  matrix: number[][];
  labels: string[];
  predicted_labels: string[];
}

interface PrecisionRecallData {
  precision: number[];
  recall: number[];
}

interface ProbabilityHistogramData {
  counts: number[];
  bin_edges: number[];
}

interface FullResults {
  model_comparison?: ModelResult[];
  feature_importance?: FeatureImportance[];
  roc_curve_data?: RocCurveData;
  confusion_matrix_data?: ConfusionMatrixData;
  precision_recall_data?: PrecisionRecallData;
  probability_histogram_data?: ProbabilityHistogramData;
}

// Cast the imported data to the FullResults type
const results: FullResults = resultsDataJson as FullResults;

export default function Home() {
  // Removed useState for loading, error, and results as data is imported directly

  // Basic check if data is available (though direct import usually guarantees it if file exists)
  if (!results) {
    return <Center style={{ height: '100vh' }}><Alert color="red" title="Error">Data could not be loaded. Check result.json.</Alert></Center>;
  }

  // Data processing logic remains the same
  const modelComparisonData = results.model_comparison?.map(item => ({
    ...item,
    Accuracy: parseFloat((item.Accuracy || 0).toFixed(3)),
    Precision: parseFloat((item.Precision || 0).toFixed(3)),
    Recall: parseFloat((item.Recall || 0).toFixed(3)),
    'F1-Score': parseFloat((item['F1-Score'] || 0).toFixed(3)),
    'ROC-AUC': parseFloat((item['ROC-AUC'] || 0).toFixed(3)),
  })) || [];

  const featureImportanceData = results.feature_importance?.slice(0, 10).map(item => ({
    ...item,
    Importance: parseFloat(item.Importance.toFixed(3)),
  })) || [];

  const rocChartData = results.roc_curve_data?.fpr.map((fpRate, i) => ({
    fpr: parseFloat(fpRate.toFixed(3)), // FPR/TPR usually don't need rounding for the plot itself, but if displayed, yes
    tpr: parseFloat((results.roc_curve_data?.tpr[i] || 0).toFixed(3)),
  })) || [];

  const prChartData = results.precision_recall_data?.recall.map((rec, i) => ({
    recall: parseFloat(rec.toFixed(3)),
    precision: parseFloat((results.precision_recall_data?.precision[i] || 0).toFixed(3)),
  })) || [];
  
  const probabilityHistogramChartData = results.probability_histogram_data?.counts.map((count, i) => ({
    bin: `${parseFloat(results.probability_histogram_data?.bin_edges[i].toString() || '0').toFixed(3)}-${parseFloat(results.probability_histogram_data?.bin_edges[i+1].toString() || '0').toFixed(3)}`,
    count: count, // Counts are integers
  })) || [];

  const titleStyle = { color: '#454545', fontSize: '18px' };
  const paperChartStyle = {
    border: '1px solid #e5e7eb', // Tailwind gray-200
    // borderRadius: '0.5rem' // Mantine default for radius="md" is usually 0.5rem. If rounded-xl was desired: 0.75rem
  };

  const DownloadIcon = (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  );

  return (
    // MantineProvider is in layout.tsx, no need here
    <Paper p="xl" shadow="xs" radius="md" style={{ maxWidth: 1200, margin: 'auto', marginTop: 20, marginBottom: 20 }}>
      <div className="flex flex-col items-center gap-2 mb-16">
        <div className="text-center font-semibold text-gray-700 text-[24px]">Model Performance Dashboard</div>
        <div className="text-center font-base text-gray-700 text-[20px]">INDENG 142A Project</div>
        <div className="text-center font-light text-gray-700 text-[18px]">JunBeum Cho</div>
        <Button 
          component="a"
          href="https://github.com/JunBeum-Cho/142A-Project/raw/refs/heads/main/dataset/conversion.csv" // Assuming the file is at public/dataset/conversion.csv
          download="conversion_data.csv" // Suggested filename for download
          variant="light"
          color='dark'
          mt="sm" // Margin top small
          leftSection={DownloadIcon}
        >
          Conversion Data (CSV)
        </Button>
      </div>
      
      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="xl">
        <Paper 
          p="md" 
          unstyled
          className="border border-gray-200 rounded-xl p-4"
        >
          <Title order={3} className='pb-10' style={titleStyle}>Model Accuracy Comparison</Title>
          {modelComparisonData.length > 0 ? (
            <BarChart
              h={400}
              data={modelComparisonData}
              dataKey="Model"
              series={[{ name: 'Accuracy', color: 'blue.6' }]}
              tickLine="xy"
              gridAxis="y"
              valueFormatter={(value) => value.toFixed(3)}
            />
          ) : <Text>No model comparison data.</Text>}
        </Paper>

        <Paper p="md" radius="md" style={paperChartStyle} shadow="none" withBorder={false}>
          <Title order={3} className='pb-10' style={titleStyle}>Feature Importances (Optimized RF - Top 10)</Title>
          {featureImportanceData.length > 0 ? (
            <BarChart
              h={featureImportanceData.length * 35 + 60}
              data={featureImportanceData.slice().reverse()}
              dataKey="Feature"
              orientation="horizontal"
              series={[{ name: 'Importance', color: 'teal.6' }]}
              xAxisLabel="Importance Score"
              gridAxis="x"
              valueFormatter={(value) => value.toFixed(3)}
            />
          ) : <Text>No feature importance data.</Text>}
        </Paper>

        {results.roc_curve_data && rocChartData.length > 0 && (
          <Paper p="md" radius="md" style={paperChartStyle} shadow="none" withBorder={false}>
            <Title order={3} className='pb-10' style={titleStyle}>ROC Curve (Optimized RF - AUC: {results.roc_curve_data.auc ? results.roc_curve_data.auc.toFixed(3) : 'N/A'})</Title>
            <LineChart
              h={300}
              data={rocChartData}
              dataKey="fpr"
              series={[{ name: 'tpr', color: 'orange.6'}]}
              curveType="linear"
              xAxisLabel="False Positive Rate"
              yAxisLabel="True Positive Rate"
              connectNulls
              valueFormatter={(value) => value.toFixed(3)} // For tooltips on line points
            />
          </Paper>
        )}

        {results.precision_recall_data && prChartData.length > 0 && (
          <Paper p="md" radius="md" style={paperChartStyle} shadow="none" withBorder={false}>
            <Title order={3} mb="md" style={titleStyle}>Precision-Recall Curve (Optimized RF)</Title>
            <LineChart
              h={300}
              data={prChartData}
              dataKey="recall"
              series={[{ name: 'precision', color: 'cyan.6'}]}
              curveType="linear"
              xAxisLabel="Recall"
              yAxisLabel="Precision"
              valueFormatter={(value) => value.toFixed(3)} // For tooltips on line points
            />
          </Paper>
        )}
        
        {results.confusion_matrix_data && (
          <Paper p="md" radius="md" style={paperChartStyle} shadow="none" withBorder={false}>
              <Title order={3} className='pb-10' style={titleStyle}>Confusion Matrix (Optimized RF)</Title>
              <Table highlightOnHover withTableBorder withColumnBorders style={{ minHeight: '75%' }}>
                  <Table.Thead>
                      <Table.Tr>
                          <Table.Th></Table.Th>
                          <Table.Th className='text-center'>{results.confusion_matrix_data.predicted_labels[0]}</Table.Th>
                          <Table.Th className='text-center'>{results.confusion_matrix_data.predicted_labels[1]}</Table.Th>
                      </Table.Tr>
                  </Table.Thead>
                  <Table.Tbody>
                      <Table.Tr>
                          <Table.Th className='text-center'>{results.confusion_matrix_data.labels[0]}</Table.Th>
                          <Table.Td className='text-center'>{results.confusion_matrix_data.matrix[0][0]}</Table.Td>
                          <Table.Td className='text-center'>{results.confusion_matrix_data.matrix[0][1]}</Table.Td>
                      </Table.Tr>
                      <Table.Tr>
                          <Table.Th className='text-center'>{results.confusion_matrix_data.labels[1]}</Table.Th>
                          <Table.Td className='text-center'>{results.confusion_matrix_data.matrix[1][0]}</Table.Td>
                          <Table.Td className='text-center'>{results.confusion_matrix_data.matrix[1][1]}</Table.Td>
                      </Table.Tr>
                  </Table.Tbody>
              </Table>
          </Paper>
        )}

        {results.probability_histogram_data && probabilityHistogramChartData.length > 0 && (
           <Paper p="md" radius="md" style={paperChartStyle} shadow="none" withBorder={false}>
              <Title order={3} className='pb-10' style={titleStyle}>Distribution of Predicted Probabilities (Optimized RF)</Title>
              <BarChart
                  h={300}
                  data={probabilityHistogramChartData}
                  dataKey="bin"
                  series={[{ name: 'count', color: 'grape.6'}]}
                  tickLine="xy"
                  gridAxis="y"
               />
          </Paper>
        )}
      </SimpleGrid>
    </Paper>
  );
}
