'use client'

import React, { useState, useEffect } from 'react';
import { Box, Container, Heading, Text, SimpleGrid, VStack, Button, Stat, StatLabel, StatNumber, StatGroup, Spinner, Table, Thead, Tbody, Tr, Th, Td } from '@chakra-ui/react';
import { useRouter } from 'next/navigation';
import { auth } from '@/utils/firebase';
import { onAuthStateChanged, User } from 'firebase/auth';
import { getUserStats, getUserDebateHistory } from '@/utils/api';

interface UserStats {
  totalDebates: number;
  remainingFreeDebates: number;
  wins: number;
  losses: number;
  draws: number;
}

interface DebateHistory {
  debateId: string;
  topic: string;
  date: string;
  result: 'win' | 'loss' | 'draw';
  userScore: number;
  aiScore: number;
}

export default function Dashboard() {
  const [user, setUser] = useState<User | null>(null);
  const [stats, setStats] = useState<UserStats | null>(null);
  const [history, setHistory] = useState<DebateHistory[]>([]);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      if (currentUser) {
        setUser(currentUser);
        try {
          const userStats = await getUserStats(currentUser.uid);
          setStats(userStats);
          const debateHistory = await getUserDebateHistory(currentUser.uid);
          
          // Round the scores in debate history
          const roundedHistory = debateHistory.map(debate => ({
            ...debate,
            userScore: Math.round(debate.userScore * 10) / 10,
            aiScore: Math.round(debate.aiScore * 10) / 10
          }));
          
          setHistory(roundedHistory);
        } catch (error) {
          console.error("Error fetching user data:", error);
        } finally {
          setLoading(false);
        }
      } else {
        router.push('/signin');
      }
    });

    return () => unsubscribe();
  }, [router]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100vh">
        <Spinner size="xl" />
      </Box>
    );
  }

  return (
    <Box bg="gray.900" minHeight="100vh" color="white">
      <Container maxW="container.xl" py={10}>
        <VStack spacing={8} align="stretch">
          <Heading as="h1" size="2xl">Welcome, {user?.displayName || 'Debater'}!</Heading>
          
          {stats && (
            <SimpleGrid columns={{ base: 2, md: 3, lg: 5 }} spacing={4}>
              <Stat bg="gray.800" p={4} borderRadius="md">
                <StatLabel>Total Debates</StatLabel>
                <StatNumber>{stats.totalDebates}</StatNumber>
              </Stat>
              <Stat bg="gray.800" p={4} borderRadius="md">
                <StatLabel>Wins</StatLabel>
                <StatNumber>{stats.wins}</StatNumber>
              </Stat>
              <Stat bg="gray.800" p={4} borderRadius="md">
                <StatLabel>Losses</StatLabel>
                <StatNumber>{stats.losses}</StatNumber>
              </Stat>
              <Stat bg="gray.800" p={4} borderRadius="md">
                <StatLabel>Draws</StatLabel>
                <StatNumber>{stats.draws}</StatNumber>
              </Stat>
              <Stat bg="gray.800" p={4} borderRadius="md">
                <StatLabel>Free Debates Left</StatLabel>
                <StatNumber>{stats.remainingFreeDebates}</StatNumber>
              </Stat>
            </SimpleGrid>
          )}

          <Box>
            <Heading as="h2" size="xl" mb={4}>Your Recent Debates</Heading>
            <Box overflowX="auto">
              <Table variant="simple">
                <Thead>
                  <Tr>
                    <Th color="gray.400">Topic</Th>
                    <Th color="gray.400">Date</Th>
                    <Th color="gray.400">Result</Th>
                    <Th color="gray.400" isNumeric>Your Score</Th>
                    <Th color="gray.400" isNumeric>AI Score</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {history.map((debate) => (
                    <Tr key={debate.debateId}>
                      <Td>{debate.topic}</Td>
                      <Td>{new Date(debate.date).toLocaleDateString()}</Td>
                      <Td color={
                        debate.result === 'win' ? 'green.400' : 
                        debate.result === 'loss' ? 'red.400' : 
                        'yellow.400'
                      }>
                        {debate.result.charAt(0).toUpperCase() + debate.result.slice(1)}
                      </Td>
                      <Td isNumeric>{debate.userScore.toFixed(1)}</Td>
                      <Td isNumeric>{debate.aiScore.toFixed(1)}</Td>
                    </Tr>
                  ))}
                </Tbody>
              </Table>
            </Box>
          </Box>

          <Button colorScheme="blue" size="lg" onClick={() => router.push('/debate/new')}>
            Start a New Debate
          </Button>
        </VStack>
      </Container>
    </Box>
  );
}