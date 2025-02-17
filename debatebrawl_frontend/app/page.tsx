'use client'

import { Box, Container, Text, Button, VStack, SimpleGrid, Flex, Heading } from '@chakra-ui/react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import Image from 'next/image';
import { useState, useEffect } from 'react';
import { auth } from '@/utils/firebase';
import { onAuthStateChanged, User, Auth } from 'firebase/auth';
import { isEmailVerified } from '@/utils/auth';
import ContinueWithGoogle from '@/components/ContinueWithGoogle';
import LoadingAnimation from '@/components/LoadingAnimation';
import { useRouter } from 'next/navigation';

const MotionBox = motion(Box as any);

interface FeatureCardProps {
  title: string;
  image: string;
  description: string;
}

const FeatureCard: React.FC<FeatureCardProps> = ({ title, image, description }) => (
  <MotionBox
    bg="gray.800"
    p={8}
    rounded="xl"
    shadow="lg"
    height="100%"
    whileHover={{ scale: 1.05 }}
    whileTap={{ scale: 0.95 }}
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5 }}
  >
    <Flex direction="column" align="center" textAlign="center">
      <Image src={image} alt={title} width={96} height={96} />
      <Text fontWeight="bold" fontSize="xl" mb={4} color="blue.400">{title}</Text>
      <Text fontSize="md">{description}</Text>
    </Flex>
  </MotionBox>
);

interface StepCardProps {
  step: string;
  title: string;
  description: string;
}

const StepCard: React.FC<StepCardProps> = ({ step, title, description }) => (
  <MotionBox
    bg="gray.800"
    p={8}
    rounded="xl"
    shadow="lg"
    position="relative"
    overflow="hidden"
    whileHover={{ scale: 1.05 }}
    whileTap={{ scale: 0.95 }}
    initial={{ opacity: 0, x: -20 }}
    animate={{ opacity: 1, x: 0 }}
    transition={{ duration: 0.5 }}
  >
    <Box position="absolute" top="-20px" left="-20px" bg="blue.500" w="80px" h="80px" rounded="full" opacity={0.2} />
    <Flex direction="column" align="flex-start">
      <Text fontSize="5xl" fontWeight="bold" color="blue.400" mb={2} zIndex={1}>
        {step}
      </Text>
      <Text fontSize="xl" fontWeight="bold" mb={4} color="white">
        {title}
      </Text>
      <Text fontSize="md" color="gray.300">
        {description}
      </Text>
    </Flex>
  </MotionBox>
);

export default function Home() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [isStartingDebate, setIsStartingDebate] = useState(false);
  const router = useRouter();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth as Auth, (currentUser) => {
      setUser(currentUser);
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  const handleStartDebate = () => {
    setIsStartingDebate(true);
    router.push('/debate/new');
  };

  if (loading) {
    return <Box>Loading...</Box>;
  }

  return (
    <Box bg="gray.900" color="white" minHeight="100vh">
      {isStartingDebate && <LoadingAnimation />}
      <Container maxW="container.xl" py={20}>
        <VStack spacing={20} align="center" textAlign="center">
          <VStack spacing={8}>
            <MotionBox
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              <Heading as="h1" fontSize={{ base: "4xl", md: "6xl", lg: "7xl" }} fontWeight="bold">
                Welcome to DebateBrawl
              </Heading>
            </MotionBox>
            <MotionBox
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5, duration: 0.8 }}
            >
              <Text fontSize={{ base: "lg", md: "xl" }} maxW="2xl">
                Challenge your intellect and debate against our advanced AI. Sharpen your skills, explore new perspectives, and become a master debater.
              </Text>
            </MotionBox>
            <MotionBox
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1, duration: 0.8 }}
            >
              {user && isEmailVerified(user) ? (
                <Button onClick={handleStartDebate} colorScheme="blue" size="lg">
                  Start a New AI Debate
                </Button>
              ) : (
                <Box width="100%" maxWidth="300px">
                  <ContinueWithGoogle />
                </Box>
              )}
            </MotionBox>
          </VStack>

          <Image
            src="/debate-illustration.svg"
            alt="AI Debate Illustration"
            width={600}
            height={400}
          />

          <VStack spacing={20} width="100%">
            <Box width="100%">
              <Heading as="h2" fontSize={{ base: "2xl", md: "3xl" }} mb={10}>
                Why Choose DebateBrawl?
              </Heading>
              <SimpleGrid columns={{ base: 1, md: 2 }} spacing={8}>
                {[
                  { title: 'Advanced AI Opponent', image: '/undraw_artificial_intelligence_re_enpp.svg', description: 'Challenge yourself against a sophisticated AI debater that adapts to your style.' },
                  { title: 'Real-time Feedback', image: '/undraw_candidate_ubwv.svg', description: 'Receive instant feedback on your arguments and debate techniques.' },
                  { title: 'Diverse Topics', image: '/undraw_firmware_re_fgdy.svg', description: 'Explore a wide range of debate topics to broaden your knowledge and skills.' },
                  { title: 'Skill Development', image: '/undraw_robotics_kep0.svg', description: 'Track your progress and improve your critical thinking and persuasion skills over time.' },
                ].map((feature, index) => (
                  <FeatureCard key={index} {...feature} />
                ))}
              </SimpleGrid>
            </Box>

            <Box width="100%" id="how-it-works">
              <Heading as="h2" fontSize={{ base: "2xl", md: "3xl" }} mb={10}>
                How It Works
              </Heading>
              <SimpleGrid columns={{ base: 1, md: 3 }} spacing={8}>
                {[
                  { step: '01', title: 'Choose Topic', description: 'Select from a variety of debate topics or suggest your own' },
                  { step: '02', title: 'Prepare', description: 'Review the topic and gather your thoughts' },
                  { step: '03', title: 'Debate', description: 'Engage in a round-by-round debate with the AI' },
                  { step: '04', title: 'Analyze', description: 'Receive a detailed analysis of your performance' },
                  { step: '05', title: 'Improve', description: "Learn from the AI's feedback and enhance your skills" },
                ].map((item, index) => (
                  <StepCard key={index} {...item} />
                ))}
              </SimpleGrid>
            </Box>
          </VStack>

          <Box textAlign="center" mt={20} mb={10}>
            <Heading as="h2" fontSize={{ base: "3xl", md: "4xl" }} mb={6}>
              Ready to Challenge Your Debate Skills?
            </Heading>
            <Text fontSize={{ base: "xl", md: "2xl" }} mb={8} maxW="2xl" mx="auto">
              Join DebateBrawl today and start your journey to becoming a master debater against AI!
            </Text>
            {user && isEmailVerified(user) ? (
              <MotionBox
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button onClick={handleStartDebate} colorScheme="blue" size="lg" fontSize="xl" py={6} px={10}>
                  Start Your AI Debate
                </Button>
              </MotionBox>
            ) : (
              <Box width="100%" maxWidth="300px" mx="auto">
                <ContinueWithGoogle />
              </Box>
            )}
          </Box>
        </VStack>
      </Container>
    </Box>
  );
}