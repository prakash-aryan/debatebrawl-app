'use client'

import { Box, Flex, Text, Button, HStack, Link as ChakraLink, useColorModeValue, useToast } from '@chakra-ui/react';
import NextLink from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { useState, useEffect } from 'react';
import { auth } from '@/utils/firebase';
import { onAuthStateChanged, User } from 'firebase/auth';
import { signInWithGoogle } from '@/utils/auth';
import { createUserDocument, getUserDocument } from '@/utils/firestore';

const NavLink = ({ children, href }: { children: React.ReactNode; href: string }) => {
  const pathname = usePathname();
  const isActive = pathname === href;

  return (
    <NextLink href={href} passHref legacyBehavior>
      <ChakraLink
        px={2}
        py={1}
        rounded={'md'}
        color={isActive ? 'blue.400' : 'white'}
        _hover={{
          textDecoration: 'none',
          bg: useColorModeValue('gray.700', 'gray.700'),
          color: 'blue.400',
        }}
        transition="all 0.3s"
      >
        {children}
      </ChakraLink>
    </NextLink>
  );
};

export default function Navbar() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(false);
  const router = useRouter();
  const toast = useToast();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
    });

    return () => unsubscribe();
  }, []);

  const handleGoogleSignIn = async () => {
    setLoading(true);
    try {
      const user = await signInWithGoogle();
      if (user) {
        const existingUser = await getUserDocument(user.uid);
        if (!existingUser) {
          await createUserDocument({
            uid: user.uid,
            email: user.email!,
            name: user.displayName || '',
            username: user.email!.split('@')[0],
            remainingFreeDebates: 2,
            totalDebates: 0,
            wins: 0,
            losses: 0,
            draws: 0,
          });
        }
        toast({
          title: 'Signed in successfully',
          status: 'success',
          duration: 3000,
          isClosable: true,
        });
        router.push('/dashboard');
      }
    } catch (error) {
      console.error('Error signing in with Google:', error);
      toast({
        title: 'Error signing in',
        description: 'An error occurred while signing in. Please try again.',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      bg={useColorModeValue('gray.900', 'gray.900')}
      px={4}
      position="fixed"
      width="100%"
      zIndex={10}
      boxShadow="0 2px 10px rgba(0,0,0,0.1)"
      bgGradient="linear(to-r, gray.900, blue.900)"
    >
      <Flex h={16} alignItems={'center'} justifyContent={'space-between'} maxW="container.xl" mx="auto">
        <NextLink href="/" passHref legacyBehavior>
          <ChakraLink _hover={{ textDecoration: 'none' }}>
            <Text fontSize="2xl" fontWeight="bold" color="blue.400" letterSpacing="wide">
              DebateBrawl
            </Text>
          </ChakraLink>
        </NextLink>

        <HStack spacing={8} alignItems={'center'}>
          <HStack
            as={'nav'}
            spacing={4}
            display={{ base: 'none', md: 'flex' }}
          >
            <NavLink href="/#how-it-works">How It Works</NavLink>
            {user ? (
              <>
                <NavLink href="/dashboard">Dashboard</NavLink>
                <NavLink href="/debate/new">New Debate</NavLink>
              </>
            ) : null}
          </HStack>
          {user ? (
            <Button
              onClick={() => auth.signOut()}
              colorScheme="blue"
              size="sm"
              fontWeight="bold"
              _hover={{
                bg: 'blue.500',
              }}
              transition="all 0.3s"
            >
              Sign Out
            </Button>
          ) : (
            <Button
              onClick={handleGoogleSignIn}
              isLoading={loading}
              colorScheme="blue"
              size="sm"
              fontWeight="bold"
              _hover={{
                bg: 'blue.500',
              }}
              transition="all 0.3s"
            >
              Get Started
            </Button>
          )}
        </HStack>
      </Flex>
    </Box>
  );
}